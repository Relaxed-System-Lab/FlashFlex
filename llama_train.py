import torch
import numpy as np
import random
import time
from llama import *
from third_party import initialize_megatron
from third_party import get_args
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, List
from hexiscale import PipelineParallel, gen_hetero_groups, validate_args, update_args
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
from flash_attn.losses.cross_entropy import CrossEntropyLoss
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def get_distributed_dataloader(dataset, bsz, shuffle = True, args = None):

    hetero_groups = get_hetero_groups()
    data_num_replicas = len(args.hetero_configs)
    pp_layouts = hetero_groups['pp_layouts']
    head_rank = []
    for tp_rank_groups in pp_layouts:
        head_rank.append(tp_rank_groups[0][0])

    assert len(bsz) == len(head_rank)

    rank = torch.distributed.get_rank()
    place_rank = 0

    for i in range(len(head_rank)):
        if head_rank[i] <= rank:
            place_rank = head_rank.index(head_rank[i])
            train_batch_size_input = bsz[i]

    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=data_num_replicas,rank=place_rank))
    return trainloader

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def forward_step_func(inputs, model):
    if isinstance(inputs, (Tuple, List)):
        outputs = model(*inputs)
    else:
        outputs = model(inputs)

    return outputs

def create_model(args):  

    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Load configs 
    llama_config = config_from_checkpoint('./llama/llama-config/', args.model_size)
    config = llama_config_to_gpt2_config(llama_config)
    overwrite_configs_and_args(config, args)
    overwrite_megatron_args(config, args)
    
    update_args(args, config)
    validate_args(args, config)

    # Generate hetero groups with respect to given config
    hetero_groups = gen_hetero_groups(pp_layouts=args.pp_layouts, hetero_configs=args.hetero_configs, layer_partitions=args.layer_partitions)
    set_hetero_groups(hetero_groups)

    hybrid_parallel_configs = get_hybrid_parallel_configs(args, hetero_groups)
    if local_rank == 0:
        print(config)
        print("=" * 80)
        print(f"Dp types: {hybrid_parallel_configs['dp_types_whole_model']}")
        print(f"Activation recompute flags: {hybrid_parallel_configs['checkpoint_flags']}")
        print(f"Global batch size per pipeline: {args.global_bsz_size}")
        print(f"Number of micro-batches per pipeline: {args.chunks}")
        print(f"Pipeline type: {args.pipeline_type}")
        print(f"Stage output saved: {not args.recompute_stage_output}")
        print(f"Gradient accumulation iter: {args.accum_iter}")
        print(f"Running iter: {args.run_iter}")
        print("=" * 80)
    
    model = construct_hybrid_parallel_model(config=config, args=args, 
                                            hybrid_parallel_configs=hybrid_parallel_configs,)
    
    # Load model checkpoints with respect to hetero_config
    state_dicts_path = "./load_model_parameters_utils/"

    # Load parameters given all layers' state dicts
    if args.load_params:
        load_model_parameters(model, config, state_dicts_path, hetero_groups, rank, args)

    # TODO: load parameters given a certain stragtegy

    loss_func = CrossEntropyLoss()

    # Initialize an optimizer

    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01) 
    elif args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return model, loss_func, optimizer

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)

    p.export_chrome_trace("/tmp/trace_" + str(torch.distributed.get_rank()) + ".json")


def is_tp_firt_rank():
    rank = dist.get_rank()
    hetero_groups = get_hetero_groups()
    tp_rank_groups = hetero_groups['tp_rank_groups']

    for tp_rank_group in tp_rank_groups:
        if rank in tp_rank_group:
            return tp_rank_group.index(rank) == 0


def train_step(model : PipelineParallel, loss_func, optimizer, trainloader, p=None):
    iter_time = 0.
    args.dataloader_length = len(trainloader)
    for idx, input_ids in enumerate(trainloader):
        args.current_iter = idx
        
        # while idx < args.train_iters:
        #     if args.profile and \
        #     idx == args.profile_step_start and \
        #     torch.distributed.get_rank() in args.profile_ranks:
        #         torch.cuda.cudart().cudaProfilerStart()
        #         torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
        
        if args.profile_mem:
            torch.cuda.memory._record_memory_history()

        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()

        input_ids_shape = [[-1, len(input_ids[0]), args.hidden_size], [-1, len(input_ids[0])], [-1, len(input_ids[0]), args.hidden_size]]

        inputs = [[input_ids], [input_ids]]
    
        # forward and backward
        if args.pipeline_type == "Gpipe":
            loss_reduced = model.gpipe_forward_backward(forward_step_func, batch=inputs, 
                                                        stage_tensor_shape=input_ids_shape, loss_func=loss_func,
                                                        args=args)
        elif args.pipeline_type == "1F1B":
            loss_reduced = model._1F1B_forward_backward(forward_step_func, batch=inputs, 
                                                        loss_func=loss_func, args=args)
        else:
            raise NotImplementedError
 
        if ((idx + 1) % args.accum_iter == 0) or (idx + 1 == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()
            if p:
                p.step()
        
        torch.cuda.synchronize()
        dist.barrier()
        end = time.time()
        iter_time += end - start

        if ((idx + 1) % args.accum_iter == 0) or (idx + 1 == len(trainloader)):
            if loss_reduced is not None and is_tp_firt_rank():
                print(f"Loss in {idx}-th iter: {loss_reduced}")
            
            tokens = args.seq_length * sum(args.global_bsz_size) * args.accum_iter
            if args.local_rank == 0:
                print(f"On rank {dist.get_rank()}, processed {tokens} tokens. Throughput in {idx} iter: {round(tokens / (iter_time), 5)} token/sec. Elapsed time: {round(iter_time, 5)}", flush=True)
                print('-' * 80)
            iter_time = 0.
        
        if args.profile_mem:
            torch.cuda.memory._dump_snapshot("./memory_utils.pickle")

        if (idx + 1) >= args.run_iter:
            # if args.local_rank == 0:
            print(f"Max allocation on rank {dist.get_rank()}: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024}")
            exit(0)
        # save_checkpoint(idx, model, optimizer)

def train(model : PipelineParallel, loss_func, optimizer, args):
    
    # model.train()

    trainloader = get_distributed_dataloader(dataset=DatasetForLlama(args), bsz=args.global_bsz_size, shuffle=True, args=args)

    for epoch in range(args.epochs):
        if args.profile:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=1,
                    active=2),
                on_trace_ready=trace_handler
            ) as p:
                train_step(model , loss_func, optimizer, trainloader, p)
        else:
            train_step(model, loss_func, optimizer, trainloader)

        
if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_arguments)
    
    args = get_args()

    set_seed()
    
    model, loss_func, optimizer = create_model(args)

    train(model, loss_func, optimizer, args)

    