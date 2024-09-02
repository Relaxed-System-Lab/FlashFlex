from cost_modeling import TimeCost, MemoryCost
from globals import configs
from arguments import get_args


args = get_args()

configs.L = args.estimate_total_layers
        
strategy_cost = TimeCost(all_pipelines=[], configs=configs)

if args.actual_running_time:
    print(f"Using provided running time, computed MFU: {round(strategy_cost.mfu(args.actual_running_time) * 100, 3)}%", )
