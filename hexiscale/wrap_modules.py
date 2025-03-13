from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, CPUOffload, MixedPrecision, BackwardPrefetch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy
import torch.nn as nn
import torch
from functools import partial
from third_party import get_args


def param_init_fn(module):
    module.to_empty(device=torch.device("cuda"))
    for m in module.modules():
        if callable(getattr(m, 'reset_parameters', None)):
            m.reset_parameters()

def param_init_fn_(module: nn.Module):
    for submodule in module.modules():
        # Handle parameters
        for param_name, param in submodule.named_parameters(recurse=False):
            if param.is_meta:
                materialized_param = nn.Parameter(
                    torch.empty_like(param, device=torch.device("cuda"))
                )
                nn.init.uniform_(materialized_param)
                setattr(submodule, param_name, materialized_param)
        # Handle buffers
        for buffer_name, buffer in submodule.named_buffers(recurse=False):
            if buffer.is_meta:
                materialized_buffer = torch.empty_like(buffer, device=torch.device("cuda"))
                # No need to apply nn.init.uniform_ unless you specifically want to for buffers.
                setattr(submodule, buffer_name, materialized_buffer)


def wrap_modules_data_parallel(module_list, dp_types, dp_groups, module_types, pp_devices=None, mixed_precision=torch.bfloat16, pp_on=True, wrap_block_name=None, root_group=None):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)
    process_group = dp_groups[0]

    # if default_process_group is not None else dp_groups[0]
    process_group = root_group if root_group is not None else process_group

    if pp_devices is not None:
        assert len(pp_devices) == len(module_list)
    for i in range(len(module_list)):
        pp_device = None if pp_devices is None else pp_devices[i]
        module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], module_type=module_types[i], pp_device = pp_device, mixed_precision=mixed_precision, pp_on=pp_on, wrap_block_name=wrap_block_name)
    

    return module_list


def wrap_data_parallel(module, dp_type = None, dp_group = None, module_type='gpt_enc', pp_device = None, mixed_precision=torch.bfloat16, pp_on=False, wrap_block_name=None):
    if dp_type is None:
        return module
    else:
        assert pp_device is not None
        # fsdp_type_dict = {0:get_args_hextrain().default_dp_type, 1:'zero3'}
        fsdp_type_dict = {0: 'ddp', 1: 'zero2', 2: 'zero3'}

        assert dp_type in fsdp_type_dict.keys(), "Unsupported dp type"

        # print(fsdp_type_dict, dp_type)
        return wrap_module_fsdp_manually(module, pp_device, module_type, dp_group, fsdp_type=fsdp_type_dict[dp_type], mixed_precision=mixed_precision, pp_on=pp_on, wrap_block_name=wrap_block_name)


def wrap_module_fsdp_manually(module, pp_device, module_type='bert_enc', dp_group=None, fsdp_type='zero3', mixed_precision=torch.bfloat16, pp_on=False, wrap_block_name=None):
    comm_group = None if dp_group is None else dp_group.group

    sharding_strategy = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}[fsdp_type]

    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision, # Param precision
        reduce_dtype=mixed_precision, # Gradient communication precision
        buffer_dtype=mixed_precision, # Buffer precision
        cast_forward_inputs=True,
        cast_root_forward_inputs=True,
        keep_low_precision_grads=False
    )
    args = get_args()

    backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE 
    # backward_prefetch = BackwardPrefetch.BACKWARD_PRE 

    fsdp_args = dict(process_group = comm_group, 
                    sharding_strategy = sharding_strategy, 
                    mixed_precision=mixed_precision_policy, 
                    backward_prefetch=backward_prefetch,
                    device_id=pp_device,
                    # use_orig_params=True,
                    param_init_fn=param_init_fn if 'initialize_on_meta' in args and args.initialize_on_meta else None,
                    limit_all_gathers=True)

    # Wrap given block
    if wrap_block_name is not None:
        if 'enc' in module_type or 'dec' in module_type:
            module = apply_fsdp(module, fsdp_args, wrap_block_name)
        else: 
            # return module
            if 'initialize_on_meta' in args and args.initialize_on_meta:
                module = FSDP(module, **fsdp_args)
            else:
                module = FSDP(module.to(pp_device), **fsdp_args)
        return module
    

def apply_fsdp(model, fsdp_args, wrap_block_name):
    check_fn=lambda submodule: (any(isinstance(submodule, block) for block in wrap_block_name))
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=FSDP,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
        **fsdp_args
    )
    return model


def wrap_modules_checkpoint(module_list, checkpoint_flags, wrap_block_name=None):
    m = module_list
    if isinstance(m, FSDP):
        m = m._fsdp_wrapped_module
    assert len(m) == len(checkpoint_flags)
    for i in range(len(m)):
        if checkpoint_flags[i]:
            if wrap_block_name is not None:
                m[i] = apply_ckpt(m[i], checkpoint_wrapper, wrap_block_name)
            else:

                m[i] = checkpoint_wrapper(m[i])
                
    return module_list


def wrap_model_checkpoint(model, wrap_block_names=[]):
    model_ = model._fsdp_wrapped_module if isinstance(model, FSDP) else model
    apply_ckpt(model_, checkpoint_wrapper, wrap_block_names)
    return model


def apply_ckpt(model, checkpoint_wrapper_fn, wrap_block_name):
    check_fn=lambda submodule: (any(isinstance(submodule, block) for block in wrap_block_name))
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True
    )
    return model
