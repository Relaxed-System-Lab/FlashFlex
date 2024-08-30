import torch
import sys
from third_party.megatron import get_args


def save_checkpoint(iter, model, optimizer):
    """Save a model checkpoint."""
    args = get_args()

    # # Only rank zero of the data parallel writes to the disk.
    # model = unwrap_model(model)

    # # Collect rng state across data parallel ranks.
    # rng_state = get_rng_state()

    # Checkpoint name.
    checkpoint_name = f"{args.hetero_configs}-{args.layer_partitions}-{torch.distributed.get_rank()}"
    exit(0)

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank() == 0:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['iteration'] = iteration
        if len(model) == 1:
            state_dict['model'] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = \
                    model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state

        # Save.
        ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()