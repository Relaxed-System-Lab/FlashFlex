from typing import Tuple
from cost_modeling import *
from globals import *


def throughput(all_pipelines: Tuple):
    """given pipelines, calculate the throughput"""
    global configs

    if all_pipelines is None:
        return 0

    model_time_cost = TimeCost(all_pipelines, configs)
    
    token_throughput = model_time_cost.token_throughput()

    return token_throughput
