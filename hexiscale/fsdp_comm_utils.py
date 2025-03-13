import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List
from dataclasses import dataclass
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _reduce_grad
from torch.distributed.fsdp._flat_param import FlatParamHandle
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.algorithms._comm_hooks.default_hooks import DefaultState
from hexiscale import CommGroup
from third_party import get_args


class DefaultState:
    r"""
    Stores state needed to perform the default communication algorithm
    within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used.
    """

    __slots__ = [
        "process_group",
        "world_size",
        "gradient_predivide_factor",
        "gradient_postdivide_factor"
    ]

    def __init__(
        self,
        process_group: dist.ProcessGroup
    ):
        if process_group is None:
            raise ValueError(f"Expected to pass in an explicit ProcessGroup to {self}.")
        self.process_group = process_group


# ==============
        # simulate [0, 2] dp, 0 has 1/2 parameters, 1 has 1/4 parameters
        # tp: [0, 1], [2, 3, 4, 5]
        # dp: [0, 2], [0, 3], [1, 4], [1, 5]
        # all_reduce: [0 * 1/2, 2], [0 * 1/2, ]
# ==============


@dataclass
class DPCommUtils(DefaultState):
    current_rank = None
    world_size = None
    tp_dim = None
    buffer_shapes = None
    buffers = []
    numels = None
    shapes = None
    param_infos = None
    comm_groups = None
    current_overall_comm_group = None

    tp_groups = None
    dp_groups = None

    layer_start = 0
    layer_end = 0
    layer_count = 0

    gradient_predivide_factor = None
    gradient_postdivide_factor = None

    cached_params = dict()
    first_exe_flag = True


    # world_size = 6, tp=2, tp=4 case
    # dp_groups = [[[0, 2], [0, 3]], [[1, 4], [1, 5]]]
    # tp_groups = [[0, 1], [2, 3, 4, 5]]

    # world_size = 3, tp=1, tp=2 case
    # dp_groups are groups for different pipelines for a single layer
    # tp_groups are groups for current tp_rank_groups in different pipeline for a single layer
    # dp_groups = [[[0, 1], [0, 2]]]
    # tp_groups = [[0], [1, 2]]

    # world_size = 2, tp=1, tp=1 case
    # dp_groups = [[[0, 1]]]
    # tp_groups = [[0], [1]]

    # world_size = 7, tp=1, tp=2, tp=4 case
    # dp_groups = [[[0, 1, 3], [0, 1, 4], [0, 2, 5], [0, 2, 6]]]
    # tp_groups = [[0], [1, 2], [3, 4, 5, 6]]

_CONFIG = DPCommUtils()

def get_fsdp_comm_config():
    global _CONFIG
    return _CONFIG

def set_fsdp_comm_config(config):
    global _CONFIG
    _CONFIG = config

def create_dist_groups(dp_groups):
    """
        calls dist.new_group to initialize all dp groups correctly
    """
    dist_groups = []
    for dp_group in dp_groups:
        for group in dp_group:
            dist_groups.append(CommGroup(group))
    return dist_groups

def get_multiplier(tp_groups, rank):
    """
        full shape of params should be current rank's params shape multiply the multiplier
    """
    multiplier = -1
    for tp_group in tp_groups:
        if rank in tp_group:
            multiplier = len(tp_group)
    return multiplier

def get_related_dp_groups(dp_groups, rank):
    related_dp_groups = []
    for dp_group in dp_groups:
        for group in dp_group:
            if rank in group:
                related_dp_groups.append(group)

    return related_dp_groups

def get_param_size(tp_groups, world_size):
    """
        Record each rank's tp size
    """
    param_size = []
    for rank in range(world_size):
        param_size.append(get_multiplier(tp_groups, rank))
    
    return param_size

def get_full_relative_sizes(tp_groups, world_size):
    """
        Find the largest tp size and dcide all other relative tp size. 
    """
    param_size = get_param_size(tp_groups, world_size)
    
    max_partition = max(param_size)

    relative_sizes = [max_partition // size for size in param_size]

    return relative_sizes

def get_current_relative_sizes(tp_groups, dp_groups, ranks, world_size):
    relative_sizes = get_full_relative_sizes(tp_groups, world_size)

    current_relative_sizes = []
    for dp_group in dp_groups:
        for group in dp_group:
            if ranks == group:
                for r in group:
                    current_relative_sizes.append(relative_sizes[r])

    return current_relative_sizes

def get_current_comm_groups(dp_groups, tp_groups, rank):
    comm_groups = get_related_dp_groups(dp_groups, rank)
    return comm_groups    

def get_full_parts(dp_groups, world_size):
    """
        Records which parts of data should each data parallel group communicate
    """
    occurence = {r: 0 for r in range(world_size)}
    full_parts = []
    for dp_group in dp_groups:
        for group in dp_group:
            part = [occurence[group[i]] for i in range(len(group))]
            for rank in group:
                occurence[rank] += 1
            full_parts.append([group, part])
    return full_parts

def get_current_part(current_group, dp_groups, world_size):
    full_parts = get_full_parts(dp_groups, world_size)  
    
    for item in full_parts:
        if item[0] == current_group:
            return item[1]


def update_param_infos(model: nn.Module, config: DPCommUtils):
    """
        By _flat_param, update necessary information for each layer on this stage.
        For each layer: 
            numels: number of elements in each tensor, 
            shapes: shape of each tensor,
            param_infos: parameter info of each tensor,
            tp_dim: records how does each tensor splitted. -1 represents no split, 0 represents ColumnParallelLinear,
                    1 represents RowParallelLinear,
            buffer_shapes: records how to create buffer during gradient synchronization

    """
    numels = []
    shapes = []
    param_infos = []
    tp_dim = []
    buffer_shapes = []
    def _recursive_apply(modules):
        for name, child in modules.named_children():
            if isinstance(child, FSDP):
                if child._flat_param is not None:
                    numels.append(child._flat_param._numels) 
                    shapes.append(child._flat_param._shapes)
                    param_infos.append(child._flat_param._param_infos)

                    dims = [-1 for _ in range(len(child._flat_param._shapes))]
                    for i in range(len(child._flat_param._shapes)):

                        shape = child._flat_param._shapes[i]
                        param_info = child._flat_param._param_infos[i]

                        if isinstance(param_info.module, RowParallelLinear):
                            dims[i] = 1
                            if shape not in buffer_shapes:
                                buffer_shapes.append(shape)
                        elif isinstance(param_info.module, ColumnParallelLinear):
                            dims[i] = 0

                    tp_dim.append(dims)

            _recursive_apply(child)
    _recursive_apply(model)

    config.numels = numels
    config.shapes = shapes
    config.param_infos = param_infos
    config.tp_dim = tp_dim
    config.buffer_shapes = buffer_shapes


def _param_not_sharded_hook(config: DPCommUtils, param: torch.Tensor,):

    """
        Conduct DDP gradient synchronization. It is a customized commmunication hook of FSDP.
        Function logic:
            1. If there is no layer has different tensor parallel size, decay to default hook to reduce the times of calling NCCL.
            2. Otherwise, check if the specific tensor is splitted. If not, synchronize them directly.
            3. Besides 1. and 2., the tensor will either be splitted by the first dimension or second dimension. 
                If the tensor is splitted in second dimension, we need to conduct transpose, copy the data to a buffer,
                conduct AllReduce, and copy back the data. This makes the data contiguous, as required by AllReduce API.
            
    """

    # print(f'entered once-{torch.distributed.get_rank()}')
    if config.layer_count == config.layer_start - 1:
        return 
    
    world_size = config.world_size
    current_rank = config.current_rank

    # while config.comm_hook_register_flags[config.layer_count] == 0:
    #     config.layer_count -= 1
    
    comm_groups = config.comm_groups[config.layer_count]
    layer_related_dp_rank_groups = config.layer_related_dp_rank_groups[config.layer_count]
    overall_comm_group = config.layer_dp_overall_comm_groups[config.layer_count]


    tp_groups = config.tp_groups[config.layer_count]
    dp_groups = config.dp_groups[config.layer_count]

    tp_dim = config.tp_dim[config.layer_count - config.layer_start]
    numels = config.numels[config.layer_count - config.layer_start]
    shapes = config.shapes[config.layer_count - config.layer_start]
    param_infos = config.param_infos[config.layer_count - config.layer_start]

    config.layer_count -= 1

    buffer_shapes = config.buffer_shapes

    # check whether this layer is symmetric
    dp_size = 1
    for i in range(len(layer_related_dp_rank_groups)):
        dp_size *= len(layer_related_dp_rank_groups[i])

    if dp_size == 1:
        for comm_group in comm_groups:
            gradient_predivide_factor, \
            gradient_postdivide_factor = update_pre_post_divide(comm_group.group)
            if gradient_predivide_factor > 1:
                param.div_(gradient_predivide_factor)
            dist.all_reduce(param, group=comm_group.group, async_op=False)
            if gradient_postdivide_factor > 1:
                param.div_(gradient_postdivide_factor)   
        return 
    
    # asymmetric all-reduce 
    overall_gradient_predivide_factor, \
    overall_gradient_postdivide_factor = update_pre_post_divide(overall_comm_group.group)

    # dim == -1 means the param is not splitted by tensor parallel and thus run overall allreduce
    param_start = 0
    for dim, numel, shape, info in zip(tp_dim, numels, shapes, param_infos):
        if dim != -1:
            param_start += numel
            continue
    
        if overall_gradient_predivide_factor > 1:
            param[param_start: param_start + numel].div_(overall_gradient_predivide_factor)
        
        # scale properly by tp-size and then sum with other ranks
        
        # for tp_group in tp_groups:
        #     if current_rank in tp_group:
        #         tp_size = len(tp_group)

        # param[param_start: param_start + numel].div_(float(tp_size))

        dist.all_reduce(param[param_start: param_start + numel], group=overall_comm_group.group)
        if overall_gradient_postdivide_factor > 1:
            param[param_start: param_start + numel].div_(overall_gradient_postdivide_factor)

        param_start += numel

    
    # otherwise smaller tp degrees will have more parameters
    for i in range(len(comm_groups)):

        comm_group = comm_groups[i]

        hetero_gradient_predivide_factor, \
        hetero_gradient_postdivide_factor = update_pre_post_divide(comm_group.group)

        # get relative sizes and parts to conduct correct all reduce communication 
        relative_sizes = get_current_relative_sizes(tp_groups, dp_groups, comm_group.ranks, world_size)
        parts = get_current_part(comm_group.ranks, dp_groups, world_size)

        # customized all reduce
        for part, relative_size, r in zip(parts, relative_sizes, comm_group.ranks):
            if r != current_rank:
                continue

            param_start = 0
            for dim, numel, shape, info in zip(tp_dim, numels, shapes, param_infos):


                if dim == -1:
                    param_start += numel
                    continue

                recovered_param = param[param_start: param_start + numel].view(shape)   
                
                size = recovered_param.size()[dim] // relative_size
                start = part * size
                end = (part + 1) * size
                
                # To make allreduce plausible
                if dim == 1:
                    # continue
                    swapped_shape = torch.Size([shape[1], shape[0]])
                    for k in range(len(buffer_shapes)):
                        if buffer_shapes[k] == shape:
                            if len(config.buffers) <= k:
                                # create buffer once
                                buffer = torch.empty(swapped_shape, dtype = param.dtype, 
                                                     device = param.device, requires_grad=False)
                                config.buffers.append(buffer)
                            else:
                                buffer = config.buffers[k]

                    buffer.copy_(torch.transpose(recovered_param, 1, 0))
                    recovered_param = buffer
                
                assert recovered_param.is_contiguous()
                if hetero_gradient_predivide_factor > 1:
                    recovered_param[start : end, ...].div_(hetero_gradient_predivide_factor)
                dist.all_reduce(recovered_param[start : end, ...], group=comm_group.group)
                if hetero_gradient_postdivide_factor > 1:
                    recovered_param[start : end, ...].div_(hetero_gradient_postdivide_factor)
  
                if dim == 1:
                    # copy back the synced param
                    param[param_start: param_start + numel].view(shape).copy_(torch.transpose(recovered_param, 1, 0))
                
                    # todo: dump into a file and inspect whether allreduce is correct
                param_start += numel
    # print(f'exited once-{torch.distributed.get_rank()}')
    return 

def _param_sharded_hook(config: DPCommUtils, padded_unsharded_grad, new_sharded_grad):

    """
        This function is the same as default hook, overwritting here is to control the frequency of gradient synchronization.
    """

    if config.layer_count == config.layer_start - 1:
        return 
    
    overall_comm_group = config.layer_dp_overall_comm_groups[config.layer_count]
    overall_gradient_predivide_factor, \
    overall_gradient_postdivide_factor = update_pre_post_divide(overall_comm_group.group)
    if overall_gradient_predivide_factor > 1:
        padded_unsharded_grad.div_(overall_gradient_predivide_factor)

    config.layer_count -= 1

    # if dist.get_rank() == 0:
    #     print(f"{overall_comm_group.ranks}-{dist.get_rank()}-{config.layer_count}-{config.current_iter}")
    dist.reduce_scatter_tensor(
        new_sharded_grad,
        padded_unsharded_grad,
        group=overall_comm_group.group,
    )

    if overall_gradient_postdivide_factor > 1:
        padded_unsharded_grad.div_(overall_gradient_postdivide_factor)

def run_hook_with_ga(func, *input):
    """
        Given a hook function, only run it per args.accum_iter. Which means gradients are synchronized according to gradient accumulation.
    """
    args = get_args()

    idx = args.current_iter
    dataloader_length = args.dataloader_length

    if ((idx + 1) % args.accum_iter == 0) or (idx + 1 == dataloader_length):
        func(*input)

    return idx, dataloader_length

def hetero_comm_ddp(config: DPCommUtils, param: torch.Tensor,):
    """
        All parameters are viewed and recovered temporarily.
    """
    run_hook_with_ga(_param_not_sharded_hook, config, param)


def reduce_scatter_hook_with_ga(config: DPCommUtils, padded_unsharded_grad, new_sharded_grad):
    run_hook_with_ga(_param_sharded_hook, config, padded_unsharded_grad, new_sharded_grad)


def update_pre_post_divide(process_group):
    def _get_gradient_predivide_factor(world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)
    
    world_size = dist.get_world_size(process_group)
    # Setting two factors `gradient_predivide_factor`
    # and `gradient_postdivide_factor` to avoid underflow and overflow
    gradient_predivide_factor = _get_gradient_predivide_factor(
        world_size
    )
    gradient_postdivide_factor = world_size / gradient_predivide_factor

    return gradient_predivide_factor, gradient_postdivide_factor


def initialize_comm_utils(hetero_groups, model: nn.Module):
    config = get_fsdp_comm_config()
    config.current_rank = dist.get_rank()
    config.world_size = dist.get_world_size()

    config.layer_start = hetero_groups['stage_idxs'][0]
    config.layer_end = hetero_groups['stage_idxs'][1]
    config.layer_count = config.layer_end - 1

    # configs that remains the same among iterations
    config.comm_groups = hetero_groups['layer_dp_comm_groups']
    config.dp_groups = hetero_groups['layer_related_dp_rank_groups']
    config.tp_groups = hetero_groups['layer_related_tp_rank_groups']
    config.layer_related_dp_comm_groups = hetero_groups['layer_related_dp_comm_groups']
    config.layer_related_dp_rank_groups = hetero_groups['layer_related_dp_rank_groups']
    config.layer_dp_overall_rank_groups = hetero_groups['layer_dp_overall_rank_groups']
    config.layer_dp_overall_comm_groups = hetero_groups['layer_dp_overall_comm_groups']

    config.dp_types_whole_model = hetero_groups['dp_types_whole_model']

    update_param_infos(model, config)

    return config
