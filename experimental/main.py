from partitioner import *
from graph import *

from evaluate import *
import time

args = get_args()
start = time.time()

npipeline = configs.npipeline

# construct with reverse of bandwidth
G = construct_graph(configs.specs[0], configs.specs[3])
G.options = configs.options

if args.apply_random_strategy:
    parts = np.random.randint(0, npipeline, size=len(configs.devices))
else:
    parts = partitioner(G, npipeline)
print("Initial partition results:", parts)

# reconstruct with bandwidth
G = construct_graph(configs.specs[0], configs.specs[1])
G.options = configs.options

next_parts = None

optimal = None
optimal_npipeline = None

for i in range(configs.niter):
    
    if i % args.log_interval == 0:
        print(npipeline)
        print(f"{i}-th iteration",)
    all_pipelines, all_sub_recovered_parts = partition_pipeline(G, parts, npipeline, i)
    if next_parts is not None:
        next_all_pipelines, next_all_sub_recovered_parts = partition_pipeline(G, next_parts, npipeline, i)
        print("Throughput", throughput(next_all_pipelines), throughput(all_pipelines))
        if throughput(next_all_pipelines) > throughput(all_pipelines):
            parts = next_parts    
            all_sub_recovered_parts = next_all_sub_recovered_parts

    if throughput(all_pipelines) > throughput(optimal):

        optimal = all_pipelines 
        optimal_npipeline = npipeline
    
    up = np.random.randint(0, 2)

    if up:
        npipeline = min(len(configs.devices) // 5, npipeline + 1)
    else:
        npipeline = max(npipeline - 1, len(configs.devices) // configs.L + 1)
    
    configs.K = initialize(configs.param, (1, npipeline))

    # construct with reverse of bandwidth
    G = construct_graph(configs.specs[0], configs.specs[3])
    G.options = configs.options

    if args.apply_random_strategy:
        parts = np.random.randint(0, npipeline, size=len(configs.devices))
    else:
        parts = partitioner(G, npipeline)

    try:
        assert npipeline == max(parts) + 1
    except AssertionError:
        print(npipeline)
        exit(0)
    # reconstruct with bandwidth
    G = construct_graph(configs.specs[0], configs.specs[1])
    G.options = configs.options



print("Output Log============================================================")
if optimal is None:
    print("Some machines will OOM, failed")
    exit(0)

optimal_simulation = TimeCost(optimal, configs)
MFU = round(optimal_simulation.mfu() * 100, 3) if optimal else 0



print(f"Optimal Throughput: {round(throughput(optimal), 3)}", )
print(f"Optimal MFU: {MFU }%", )
print(f'Optimal time: {round(optimal_simulation.overall_cost(), 3)}', )
print(f"N-pipeline: {optimal_npipeline}")


# For detailed logs:
if args.verbose:
    print("Optimal Placement:", )
    for id, pipeline in enumerate(optimal):
        print(f"    {id}-th pipeline: {[len(stage) for stage  in pipeline[0]]}", '\n',
              f"    - devices: {[[configs.devices[device_id].machine_id for device_id in stage] for stage in pipeline[0]]}", '\n',
              f"    - devices name: {[[configs.devices[device_id].name for device_id in stage] for stage in pipeline[0]]}", '\n',
              f"    - layer partitions: {pipeline[1]}", '\n',
              f"    - memory estimation: {pipeline[2]}", '\n',
               )

end = time.time()
print("=" * 80)
print("Consumed Time(s):", round(end-start, 3))
