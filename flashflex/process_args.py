import torch
import torch.distributed as dist
import json

def update_args(args, config):


    assert int(args.all_hetero_configs_path is None) + int(args.hetero_configs is None) == 1, "Choose one way to start the program"

    if args.all_hetero_configs_path is not None:
        with open(args.all_hetero_configs_path, 'r') as all_hetero_configs_file:
            all_hetero_configs = json.load(all_hetero_configs_file)
            args.hetero_configs = all_hetero_configs['hetero_configs']
            args.layer_partitions = all_hetero_configs['layer_partitions']

            if 'pp_layouts' in all_hetero_configs.keys():
                args.pp_layouts = all_hetero_configs['pp_layouts']
    else:
        args.layer_partitions = eval(args.layer_partitions)
        args.hetero_configs = eval(args.hetero_configs)
        if args.pp_layouts is not None:
            args.pp_layouts = eval(args.pp_layouts)

    args.lr = args.learning_rate

    args.seq_length = 4096 
    config.n_positions = args.seq_length
    args.num_hidden_layers = args.total_layer_num
    config.n_layer = args.num_hidden_layers

def validate_args(args, config):
    hetero_configs = args.hetero_configs
    layer_partitions = args.layer_partitions
    total_layer_num = args.total_layer_num
    global_bsz_size = args.global_bsz_size
    chunks = args.chunks

    world_size = dist.get_world_size()
    
    num_heads = config.n_head

    for pipeline in hetero_configs:
        for tp_size in pipeline:
            assert num_heads % tp_size == 0, "Num heads must be divisible by tp size"


    assert world_size == sum([sum(pipeline_ranks) for pipeline_ranks in hetero_configs]), 'Wrong hetero configs'
    for i, layer_partition in enumerate(layer_partitions):
        assert total_layer_num == sum(layer_partition), f'Wrong layer partition for pipeline {i}'

    length_gap = len(hetero_configs) - len(global_bsz_size)
    if length_gap > 0:
        padding_global_bsz = args.global_bsz_size[-1]
        padding_chunks = args.chunks[-1]

        for _ in range(length_gap):
            args.global_bsz_size.append(padding_global_bsz)
            args.chunks.append(padding_chunks)

    assert len(global_bsz_size) == len(chunks) == len(hetero_configs), 'Wrong length of globl batch size or chunks, should be the same as number of pipelines'

    assert args.run_iter >= args.accum_iter, 'Not enough iterations for one gradient accumulation cycle'

    # if args.checkpoint_layers:
    #     assert args.checkpoint_all, "Currently only support activation recompute on all layers"


    for hetero_config, layer_partition in zip(hetero_configs, layer_partitions):
        assert len(hetero_config) == len(layer_partition), "Hetero config should have the same length as layer partition"

    
    if args.pp_layouts is not None:
        sorted_pp_layouts = []
        for pp_layout, hetero_config in zip(args.pp_layouts, hetero_configs):
            sorted_pp_layout = []
            for stage_layout, stage_length in zip(pp_layout, hetero_config):
                assert len(stage_layout) == stage_length, "Wrong pp_layouts, each stage should contain correct number of devices based on hetero_configs"
                sorted_pp_layout.append(list(sorted(stage_layout)))
            sorted_pp_layouts.append(sorted_pp_layout)
        
        args.pp_layouts = sorted_pp_layouts
        
