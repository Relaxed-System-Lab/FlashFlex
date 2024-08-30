"""
    This script helps to generate parallel --hetero_configs, --layer_partitions for baseline"

    assumes a node has 8 GPUs

    --hetero_configs [[4, 4, 4, 4], [4, 4, 4, 4]]
    --layer_partitions [[20, 20, 20, 20], [20, 20, 20, 20],]
"""

import argparse
import os
import numpy as np
import json

from collections import OrderedDict

def generate_cuda_id_mapping(devices_ids):

    rank_cuda_id_mapping = {}

    devices_counts = OrderedDict()
    for pipeline_ids in devices_ids:
        for stage_ids in pipeline_ids:
            if stage_ids[0] not in devices_counts:
                devices_counts[stage_ids[0]] = len(stage_ids)
            else:
                devices_counts[stage_ids[0]] += len(stage_ids)

    used_devices = list(devices_counts.values())
    devices_index = list(devices_counts.keys())

    rank = 0
    for pipeline_ids in devices_ids:
        for stage_ids in pipeline_ids:
            for id in stage_ids:
                rank_cuda_id_mapping[rank] = devices_counts[id] - used_devices[devices_index.index(id)]
                used_devices[devices_index.index(id)] -= 1
                rank += 1
    
    return rank_cuda_id_mapping


def get_store_path():

    parent_path = f'./llama-scripts-logs'
    store_path = f'{parent_path}/{args.model_size}-scripts'
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)

    if not os.path.exists(store_path):
        os.mkdir(store_path)
    else:
        file_list = os.listdir(store_path)
        for file in file_list:
            file_path = os.path.join(store_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return store_path


def update_args(args):

    assert int(args.all_hetero_configs_path is None) + int(args.hetero_configs is None) == 1, "Choose one way to start the program"

    if args.all_hetero_configs_path is not None:
        with open(args.all_hetero_configs_path, 'r') as all_hetero_configs_file:
            all_hetero_configs = json.load(all_hetero_configs_file)
            args.hetero_configs = all_hetero_configs['hetero_configs']
            args.layer_partitions = all_hetero_configs['layer_partitions']
            args.devices_id = all_hetero_configs['devices_id']
    else:

        args.hetero_configs = eval(args.hetero_configs)
        args.layer_partitions = eval(args.layer_partitions)
        args.devices_id = eval(args.devices_id)

    args.world_size = sum([sum(stage_size for stage_size in pipeline) for pipeline in args.hetero_configs])


def validate_args(args):
    hetero_configs = args.hetero_configs
    layer_partitions = args.layer_partitions
    devices_id = args.devices_id

    for i, layer_partition in enumerate(layer_partitions):
        assert args.layer_num == sum(layer_partition), f'Wrong layer partition for pipeline {i}'

    assert len(hetero_configs) == len(layer_partitions) == len(devices_id), "unmatched length"

    for hetero_config, device_id in zip(hetero_configs, devices_id):
        for stage_config, stage_device_id in zip(hetero_config, device_id):
            assert stage_config == len(stage_device_id), "devices id or hetero configs is wrong"
        

def create_script(args, store_path, rank_cuda_id_mapping):

    hetero_configs = args.hetero_configs
    layer_partitions = args.layer_partitions

    npipelines = len(hetero_configs)
    input_bsz_len = len(args.global_batch_size)

    cuda_id = rank_cuda_id_mapping[args.rank]


    if npipelines > input_bsz_len:
        for _ in range( npipelines - input_bsz_len ):
            print(f"assume {_ + 1}-th pipeline has the same batch size as { input_bsz_len - 1}-th")
            args.global_batch_size.append(args.global_batch_size[-1])
            args.micro_batch_num.append(args.micro_batch_num[-1])
    else:
        args.global_batch_size = args.global_batch_size[:npipelines]
        args.micro_batch_num = args.micro_batch_num[:npipelines]

    global_batch_size_str = " ".join(args.global_batch_size)
    micro_batch_num_str = " ".join(args.micro_batch_num )

    rank_script_path = f'{store_path}/r{args.rank}.sh'

    with open(rank_script_path, 'w' ) as f:
        if args.nccl_socket_ifname is not None:
            f.write(f"export NCCL_SOCKET_IFNAME={args.nccl_socket_ifname}")
        f.write("export NCCL_IB_DISABLE=0  \n")
        f.write("export NCCL_IB_HCA=mlx5_2,mlx5_5   \n")
        f.write(f"export MASTER_ADDR={args.master_addr} \n")
        f.write(f"export MASTER_PORT={args.master_port} \n")
        f.write(f"export WORLD_SIZE={args.world_size} \n")
        f.write(f"export RANK={args.rank} \n")

        f.write(f"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES={cuda_id} python3 llama_train.py ")

        f.write(f"--model_size {args.model_size}  ")
        f.write("--fp16 ")
        f.write("--mixed_precision fp16 ")
        f.write("--use-flash-attn ")
        f.write(f"--hetero_configs \"{str(hetero_configs)}\"    ")
        f.write(f"--layer_partitions \"{str(layer_partitions)}\"   ")
        f.write(f"--total-layer-num {str(args.layer_num)}    ")
        f.write(f"--chunks {micro_batch_num_str}    " )
        f.write(f"--global_bsz_size {global_batch_size_str}    " )
        f.write(f"--checkpoint-layers  ")
        f.write(f"--checkpoint-all  ")
        f.write(f"--accum-iter {args.accum_iter} ")
        f.write(f"--run-iter {args.run_iter}  ")
        if args.recompute_stage_output:
            f.write(f"--recompute-stage-output ")

def run_file_list(args):
    devices_id = args.devices_id    

    run_file_list = []

    retain_rank = 0

    for pipeline_devices in devices_id:
        for stage_devices in pipeline_devices:
            for device in stage_devices:
                if device == args.current_device:
                    run_file_list.append(retain_rank)
                retain_rank += 1
    
    return run_file_list


parser = argparse.ArgumentParser()
parser.add_argument('--model-size', type=str, default='llama-30b')
parser.add_argument('--master_addr', type=str, default='localhost')
parser.add_argument('--master_port', type=str, default='9991')
parser.add_argument('--layer-num', type=int, default=60)
parser.add_argument('--current_device', type=int, default=0)
parser.add_argument('--micro_batch_num', type=str, nargs='+', default='1')
parser.add_argument('--global_batch_size', type=str, nargs='+', default='1')
parser.add_argument('--hetero_configs', type=str,  default=None)
parser.add_argument('--layer_partitions', type=str, default=None)
parser.add_argument('--devices_id', type=str, default=None)
parser.add_argument('--accum-iter', type=int, default=1)
parser.add_argument('--run-iter', type=int, default=10)
parser.add_argument('--retain-run-file', action='store_true', )
parser.add_argument('--recompute-stage-output', action='store_true', )
parser.add_argument('--nccl_socket_ifname', type=str, default=None)
parser.add_argument('--all_hetero_configs_path', type=str, default=None, help="JSON file path to hetero configs")
    

if __name__ == "__main__":
    args = parser.parse_args()

    update_args(args)

    validate_args(args)

    store_path = get_store_path()

    rank_cuda_id_mapping = generate_cuda_id_mapping(args.devices_id)

    create_list = run_file_list(args) if args.retain_run_file else [i for i in range(args.world_size)]

    for i in create_list:
        args.rank = i    
        create_script(args, store_path, rank_cuda_id_mapping)

