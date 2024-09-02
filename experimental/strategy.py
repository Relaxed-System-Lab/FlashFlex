from globals import configs
import numpy as np
from cost_modeling import MemoryCost, TimeCost
from typing import List
from arguments import get_args
from copy import deepcopy

def get_considered_stragegies(i):
    """
        Do not consider stages over 3
    """
    args = get_args()
    considered_strategies = {}
    if args.not_use_tp:

        considered_strategies[1] = [[1]]
        considered_strategies[2] = [[1, 1]]
        considered_strategies[3] = [ [1, 1, 1]]
        considered_strategies[4] = [[1, 1, 1, 1]]
        considered_strategies[5] = [[1, 1, 1, 1, 1]]
        considered_strategies[6] = [[1, 1, 1, 1, 1, 1]]
        considered_strategies[7] = [[1, 1, 1, 1, 1, 1, 1]]
        considered_strategies[8] = [[1, 1, 1, 1, 1, 1, 1, 1] ]
    
    else:
        considered_strategies[1] = [[1]]
        considered_strategies[2] = [[2], [1, 1]]
        considered_strategies[3] = [[2, 1], [1, 1, 1]]
        considered_strategies[4] = [[4], [2, 2], [2, 1, 1]]
        considered_strategies[5] = [[4, 1], [2, 2, 1]]
        considered_strategies[6] = [[4, 2], [2, 2, 2]]
        considered_strategies[7] = [[4, 2, 1], [2, 2, 2, 1]]
        considered_strategies[8] = [[8], [4, 4], [4, 2, 2], ]

    return considered_strategies[i]

def refine_strategies(strategies, pp_devices):

    global configs

    ngpu_strategy = []

    def tp_first():
        for possible_strategies in strategies:
            shortest = [0] * 100
            for possible_strategy in possible_strategies:
                
                shortest = possible_strategy if len(possible_strategy) < len(shortest) else shortest
            ngpu_strategy.extend(shortest)

    def pp_first():
        for possible_strategies in strategies:
            shortest = [0] * 1
            for possible_strategy in possible_strategies:
               
                shortest = possible_strategy if len(possible_strategy) >= len(shortest) else shortest
            ngpu_strategy.extend(shortest)      
        

    def tp_pp_tradeoff():
        global configs

        offsets: List = np.cumsum([sum(possible_strategy[0]) for possible_strategy in strategies]).tolist()
        offsets.insert(0, 0)

        for i in range(len(strategies)):
            local_optimal = None
            local_minimum_cost = 1e8
            possible_strategies = strategies[i]
            for stage_strategy in possible_strategies:
                strategy_offsets: List = np.cumsum(stage_strategy).tolist()
                strategy_offsets.insert(0, 0)

                stage = [pp_devices[offsets[i]: offsets[i + 1]][strategy_offsets[j]: strategy_offsets[j + 1]] for j in range(len(strategy_offsets) - 1)]

                # simulate by fake layer partition
                stage_layer_partition = np.round([configs.L * ndevices // sum(stage_strategy) for ndevices in stage_strategy], decimals=0)
                stage_layer_partition[-1] -= sum(stage_layer_partition) - configs.L
                assert sum(stage_layer_partition) == configs.L

                strategy_cost = TimeCost(all_pipelines=[[stage, stage_layer_partition]], configs=configs)

                # update strategy_cost to 1e8 if oom
                if local_minimum_cost > strategy_cost.pipeline_cost(pp_id=0):
                    local_minimum_cost = strategy_cost.pipeline_cost(pp_id=0)
                    local_optimal = stage_strategy

            ngpu_strategy.extend(local_optimal)
                
    tp_pp_tradeoff()

    return ngpu_strategy


def gen_strategy(recovered_parts, path):
    global configs

    recovered_parts = list(sorted(recovered_parts, key=lambda ele: path[recovered_parts.index(ele)]))

    # recovered_parts = [[1, 3, 6, 8, 9], [2, 4, 11, 12]]
    strategies = []

    parts_machines = [[configs.device_machine_map[gpu] for gpu in gpus] for gpus in recovered_parts]
    intra_counts = [[parts_machine.count(i) for i in set(parts_machine)] 
                    for parts_machine in parts_machines]
    
    for stage_counts in intra_counts:
        for ngpus in stage_counts:
            strategies.append(get_considered_stragegies(ngpus))

    
    # like [[0, 1], [2], [3]]
    flattened_parts = [recovered_parts[i][j] for i in range(len(recovered_parts)) for j in range(len(recovered_parts[i]))]

    # like [[2, 1], [1], [1]]
    ngpu_strategy = refine_strategies(strategies, flattened_parts)
    
    strategy = []
    i = 0
    start = 0
    while i < len(ngpu_strategy):
        
        end = start + ngpu_strategy[i]
        strategy.append(flattened_parts[start : end])

        start = end
        i += 1

    strategy_machine = [configs.device_machine_map[stage_device[0]] for stage_device in strategy]
    
    return strategy