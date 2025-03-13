import os
import sys
sys.path.insert(0, '..')
import torch
from torch import nn
import numpy as np
from typing import Any
from hexiscale import PipelineParallel, PipeSequential
from flash_attn.modules.block import Block
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from ..arguments import get_hetero_groups
from third_party import get_args
from collections import OrderedDict
from .Llamamodel_pipeline import LlamaEmbeddings_, LlamaLayers_, LlamaPreNorm_, LlamaCls_
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.gpt import create_mixer_cls, create_mlp_cls


def display(hetero_groups, separator,):
    args = get_args()
    def _show_on_rank(show_rank, line_idx):
        if dist.get_rank() == show_rank and (not args.display_one_pipeline or args.local_rank == 0):
            layer_related_dp_rank_groups, tp_rank_groups, \
                pp_rank_groups, pp_ranks_whole_model, \
                    p2p_lists = hetero_groups['layer_related_dp_rank_groups'], hetero_groups['tp_rank_groups'], \
                                hetero_groups['pp_rank_groups'], hetero_groups['pp_ranks_whole_model'], hetero_groups['p2p_lists'] 
            
            forward_lists, backward_lists = p2p_lists

            def format_list(lst):
                return '\n'.join(['    - ' + str(item) for item in lst])

            def format_ranks(tp_rank_groups):
                max_length = 0
                for tp_rank_group in tp_rank_groups:
                    max_length = max(len(tp_rank_group), max_length)

                place_holder = "x"
                print_lists = []
                
                for i in range(max_length):
                    print_list = []
                    for tp_rank_group in tp_rank_groups:
                        if len(tp_rank_group) > i:
                            print_list.append(str(tp_rank_group[i]))
                        else:
                            print_list.append(place_holder)
                    print_lists.append("  ".join(print_list))
                

                return format_list(print_lists)
            
            def print_lists(lsts, lsts_type):
                send_list, recv_list, send_empty_list, recv_empty_list = lsts
                print(f'P2p Lists in {lsts_type}'.center(80))
                print('Send List:', send_list)
                print('Recv List:', recv_list)
                print('Send Empty List:', send_empty_list)
                print('Recv Empty List:', recv_empty_list)

            pp_separator = '-' * 80


            print(separator)
            print(f'Heterogeneous Parallel Configuration on pipeline {line_idx}'.center(80))
            print(separator)

            print(format_ranks(tp_rank_groups), sep='')
            print(separator)

            print('Splitting Layers to Pipeline Parallel Stages'.center(80))
            print('    - ', pp_ranks_whole_model)
            print(separator)
        
            def find_first_indices(lst):
                indices = {}
                result = []
                for i, value in enumerate(lst):
                    if value not in indices:
                        indices[value] = i
                        # Subtract 1 from the index for all values except for 0
                        result.append(i if value == 0 else i-1)
                return result
            pp_indices = find_first_indices(pp_ranks_whole_model)

            print('The First PP Layer Index of Each Stage'.center(80))
            print(pp_separator)
            print('    - List:', pp_indices)
            print(separator)

    def print_dp(layers_dp_rank_groups):
        dict_count = OrderedDict()
        for layer_dp_rank_groups in layers_dp_rank_groups:
            for layer_dp_rank_group in layer_dp_rank_groups:
                for rank_group in layer_dp_rank_group:
                    group = tuple(rank_group)
                    if group not in dict_count:
                        dict_count[group] = 1
                    else:
                        dict_count[group] += 1
            
        for key, val in dict_count.items():
            print(f'    - group: {key}, layers: {val}')
    
    dp_separator = '-' * 80

    if torch.distributed.get_rank() == 0:
        print(separator)
        print('Data Parallel Groups'.center(80))
        print(dp_separator)
        print_dp(hetero_groups['layer_related_dp_rank_groups'])

    torch.distributed.barrier()
    pp_layouts = hetero_groups['pp_layouts']
    
    for line_idx in range(len(pp_layouts)):
        
        _show_on_rank(show_rank=pp_layouts[line_idx][0][0], line_idx=line_idx)
        
def get_current_chunks(chunks, pp_layouts):
    assert len(chunks) == len(pp_layouts), "Each pipeline must assign a microbatch num"
    for chunk, pp_layout in zip(chunks, pp_layouts):
        rank = dist.get_rank()
        for group in pp_layout:
            if rank in group:
                return chunk

def construct_hybrid_parallel_model(
        config,
        args,
        hybrid_parallel_configs,
        ):

    """
    Constructs a hybrid parallel model for pretraining large-scale models Llama.
    This function integrates various parallelism techniques, including tensor parallelism (TP), 
    pipeline parallelism (PP), and data parallelism (DP) to efficiently distribute the model across 
    multiple devices and nodes.

    Args:
        config: Configuration parameters specific to the model architecture.
        args: Training args.
        hybrid_parallel_configs: Configuration parameters for hybrid parallelism, including degrees of TP and PP.

    The function performs the following steps:
    1. Extracts and sets up the required configurations for different types of parallelism.
    2. Initializes the configurations for TP, PP and DP across the entire model.
    3. Generates heterogeneous groups and communication lists for effective model parallelism.
    4. Constructs the model layers using specialized Llama modules.
    5. Defines the output tensor shapes, data types, and sizes for each model layer under the parallelism setup.
    6. Assembles the final hybrid parallel model ready for training.

    Returns:
        A PipelineParallel model that integrates TP, PP, DP for efficient large-scale model pretraining.
    """

    # Generate the model 
    if args.local_rank == 0:
        print("Creating Model...")

    hetero_groups = get_hetero_groups()

    # Initialize on cuda in training version
    gpt_model = GPTLMHeadModel(config, device='meta' if args.initialize_on_meta else 'cuda', dtype=args.params_dtype)
    
    factory_kwargs = {'device': 'meta' if args.initialize_on_meta else 'cuda', 'dtype': args.params_dtype,}

    for i in range(config.num_hidden_layers):
        layer = gpt_model.transformer.layers[i]
        setattr(layer, 'mixer', create_mixer_cls(config, layer_idx=i, process_group=hetero_groups['current_tp_group'], **factory_kwargs)(config.hidden_size))
        setattr(layer, 'mlp', create_mlp_cls(config, layer_idx=i, process_group=hetero_groups['current_tp_group'], **factory_kwargs)(config.hidden_size))

    hp_configs = hybrid_parallel_configs

    dp_types_whole_model = hp_configs['dp_types_whole_model']
    
    p2p_lists, pp_ranks_whole_model = hetero_groups['p2p_lists'], hetero_groups['pp_ranks_whole_model']
    
    separator = '=' * 80
    display(hetero_groups, separator)
        
    model = PipeSequential()

    embeddings = LlamaEmbeddings_(gpt_model)
    prenorm = LlamaPreNorm_(gpt_model)
    cls = LlamaCls_(gpt_model)

    model.add_module('embeddings', embeddings)
    for i in range(config.num_hidden_layers):
        enc = LlamaLayers_(gpt_model, i, i + 1)
        model.add_module('layer_%d'%i, enc)
    model.add_module('prenorm', prenorm)
    model.add_module('cls', cls)

    seq_len, hidden_size = args.seq_length, args.hidden_size
    layer_output_tensor_shapes = [None] + [[[-1,seq_len,hidden_size],]] * config.num_hidden_layers + [None] * 2
    mixed_precision = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.mixed_precision]
    layer_output_tensor_dtypes = [None] + [[mixed_precision,]] * config.num_hidden_layers + [None] * 2
    
    layer_dp_sizes = [len(lst) for lst in hetero_groups['layer_related_tp_rank_groups']]

    # Some hints
    # pp_ranks_whole_model = [0] + [0,1,2] + [2,2]
    # pp_groups = [[0,2,4],[1,3]]
    # pp_groups = [[0,2,3],[1,2,4],[0,2,3],[0,2,3],[1,2,4]]
    # broadcast_group = dist.new_group([3,4])

    current_chunks = get_current_chunks(args.chunks, hetero_groups['pp_layouts'])

    hp_model = PipelineParallel(
                model = model, 
                model_ranks = pp_ranks_whole_model, 
                all_lines_model_ranks = hetero_groups['all_pp_ranks_whole_model'],
                layer_output_tensor_shapes = layer_output_tensor_shapes, 
                layer_output_tensor_dtypes = layer_output_tensor_dtypes,
                layer_dp_sizes = layer_dp_sizes, 
                chunks=current_chunks,
                p2p_lists = p2p_lists,
                process_group = hetero_groups['current_pp_rank_groups'],
                broadcast_group = hetero_groups['current_tp_group'],
                broadcast_group_list = hetero_groups['current_tp_rank_group'],
                info=False,
                show_process=False)
    
    module_types = ['embed'] + ['gpt_dec'] * config.num_hidden_layers + ['norm', 'cls']
    

    hp_model.wrap_pipeline_modules_data_parallel(
            dp_types_whole_model,
            hetero_groups['layer_dp_overall_comm_groups'],
            module_types=module_types,
            mixed_precision=mixed_precision,
            wrap_block_name=[Block],
            )    

    hp_model.wrap_modules_data_parallel_comm_hook(hetero_groups, args, dp_types_whole_model)

    hp_model.wrap_pipeline_modules_checkpoint(hp_configs['checkpoint_flags'], wrap_block_name=None)

    return hp_model


def get_hybrid_parallel_configs(args, hetero_groups):
    layer_related_tp_rank_groups = hetero_groups['layer_related_tp_rank_groups']

    dp_types_whole_model = []

    for tp_rank_groups in layer_related_tp_rank_groups:
        default_dp_type = 1
        tp_size = len(tp_rank_groups[0])
        for rank_group in tp_rank_groups:
            if tp_size == len(rank_group):
                continue
            # different tp size occurs, only apply ddp
            else:  
                default_dp_type = 0
                break
        dp_types_whole_model.append(default_dp_type)

    num_layers = args.num_hidden_layers

    # embedding, transformer layers, prenorm, cls
    if args.default_dp_type:
        fsdp_type_dict = {'ddp':0, 'zero2': 1,'zero3':2}
        dp_types_whole_model = [fsdp_type_dict[args.default_dp_type]] * (num_layers + 3)

    hetero_groups['dp_types_whole_model'] = dp_types_whole_model

    
    checkpoint_flags = [1 if args.checkpoint_all else 0] \
          + [1 if args.checkpoint_layers else 0 for _ in range(num_layers)] \
          + [1 if args.checkpoint_all else 0 for _ in range(2) ]
    hybrid_parallel_configs = {
        'dp_types_whole_model': dp_types_whole_model,
        'checkpoint_flags': checkpoint_flags,
    }
        
    return hybrid_parallel_configs

def overwrite_megatron_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.max_position_embeddings = config.max_position_embeddings
    args.use_cpu_initialization = True
