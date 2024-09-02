from cost_modeling import TimeCost, MemoryCost
from globals import configs
from arguments import get_args


args = get_args()

configs.L = args.estimate_total_layers

strategies = args.estimate_strategy
layer_partitions = args.estimate_layer_partition

if args.strategy_device_ids is not None:
    strategy_device_ids = eval(args.strategy_device_ids)

print(layer_partitions, configs.L)
assert sum(layer_partitions[0]) == configs.L


all_pipelines = []
all_pipelines_mem = []
if args.strategy_device_ids is None:
    rank = 0
    for strategy, layer_partition in zip(strategies, layer_partitions):
        strategy_device_ids = []
        pipeline_mem = []
        for stage_idx, (stage_length, nlayers) in enumerate(zip(strategy, layer_partition)):
            stage_device_ids = []
            for _ in range(stage_length):
                stage_device_ids.append(rank)
                rank += 1
            strategy_device_ids.append(stage_device_ids)
            flags = [1 if stage_idx == 0 else 0] + [1 if stage_idx == len(strategy) - 1 else 0] * 2
            stage_mem = MemoryCost(device_memory=None, layers=nlayers, stage_strategy=strategy, configs=configs, flags=flags)
            pipeline_mem.append(stage_mem)
        all_pipelines.append([strategy_device_ids, layer_partition, ''])
        all_pipelines_mem.append(pipeline_mem)
else:
    for strategy_device_id, layer_partition in zip(strategy_device_ids, layer_partitions):
        all_pipelines.append([strategy_device_id, layer_partition, ''])

        
print(all_pipelines)
# exit(0)
strategy_cost = TimeCost(all_pipelines=all_pipelines, configs=configs)
print("Estimation Input Log============================================================")
print("All pipelines (strategy, layer partition, memory estimation):", all_pipelines)
print("Time Cost Estimation", "=" * 59)
print(f"Batch size per pipeline: {configs.B}, Micro batch size per pipeline: {configs.MB}")
print("Time Cost:", strategy_cost.overall_cost())
print(f"DP time cost: {strategy_cost.dp_cost()}")
print(f"Processed tokens: {configs.T}")
print("Throughput:", strategy_cost.token_throughput())
print(f"MFU: {round(strategy_cost.mfu() * 100, 3)}%", )

if args.actual_running_time:
    print(f"Using provided running time, computed MFU: {round(strategy_cost.mfu(args.actual_running_time) * 100, 3)}%", )


if not args.estimate_all:
    exit(0)
print("Mem Cost Estimation", "=" * 60)

for pp_id, (pipeline_mem, strategy) in enumerate(zip(all_pipelines_mem, strategies)):
    for stage_id, (mem, stage_length) in enumerate(zip(pipeline_mem, strategy)):
        print(f"For {pp_id}-th pipeline, {stage_id}-th stage (pipeline strategy: {strategy}), \n estimeated overall memory cost: {mem.overall_memory() * stage_length },")
        if args.verbose:
            print(f"parameter memory cost: {mem.param_memory()}, activation memory: {mem.activation_memory(recompute=True)}, per device memory cost: {mem.overall_memory(recompute=True)}")

