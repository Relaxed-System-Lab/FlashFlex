import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-size', type=str, default='llama-7b',
                        help='Instruct which model to work on')
    parser.add_argument('--npipeline', type=int, default=1,
                        help="Instruct how many pipelines to be created")
    parser.add_argument('--inter_bw', type=float, default=5,
                        help='Assumed inter-machine bandwidth')
    parser.add_argument('--global_bsz', type=int, default=1,
                        help="Per pipeline global-batch size")
    parser.add_argument('--micro_bsz', type=int, default=1,
                        help="Per pipeline micro-batch size")
    parser.add_argument('--MB', type=int, nargs="+", default=None,
                        help="Per pipeline micro-batch size")
    parser.add_argument('--kway', type=int, default=3,
                        help='Instruct greedy path finding')
    parser.add_argument('--recompute', type=bool, default=True,
                        help='If enabled, activation recompute is considered')
    parser.add_argument('--estimate_strategy', type=str, default='[]',
                        help='Pipeline Strategies to be estimated')
    parser.add_argument('--strategy_device_ids', type=str, default=None,
                        help='Pipeline Strategies to be estimated')
    parser.add_argument('--estimate_layer_partition', type=str, default='[]',
                        help='Layer partitions to be estimated')
    parser.add_argument('--estimate_total_layers', type=int, default=32,
                        help='Adjust when running different total layers from standard model')
    parser.add_argument('--accum-iter', type=int, default=1,
                        help='Adjust batch size for more accurate time cost estimation')
    parser.add_argument('--verbose', type=bool, default=False,
                        help="If enabled, outputs will be in detail")
    parser.add_argument('--actual_running_time', type=float, default=None,
                        help='If provided, a MFU will be computed by it')
    parser.add_argument('--estimate_all', action='store_true', 
                        help='If enabled, both memory and time cost are estimated, otherwise only time cose is estimated')
    parser.add_argument('--machine_config_path', type=str, default=None, 
                        help='If enabled, both memory and time cost are estimated, otherwise only time cose is estimated')
    parser.add_argument('--not_use_tp', action='store_true', 
                        help='Whether consider tp')
    parser.add_argument('--zero_3', action='store_true', 
                        help='Whether consider zero-3')
    parser.add_argument('--log_interval', type=int, 
                        help='Log interval')
    parser.add_argument('--niter', type=int, 
                        help='niter')
    parser.add_argument('--apply_random_strategy', action='store_true', 
                        help='random strategy instead of graph partition')


    args = parser.parse_args()

    args.estimate_strategy = eval(args.estimate_strategy)
    args.estimate_layer_partition = eval(args.estimate_layer_partition)

    return args

