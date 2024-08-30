import argparse

def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='hexgen arguments')

    # hetro parallelism arguments
    group.add_argument(
        "--local-rank", type=int, default=-1, help="Local rank.",
    )
    parser.add_argument(
        "--model_size", type=str, default='llama-7b', help="Model size.", choices=['llama-7b', 'llama-13b', 'llama-30b', 'llama-70b']
    )
    parser.add_argument(
        "--overwrite_config", type=int, default=0, help="Whether to overwrite model config"
    )
    group.add_argument(
        "--initialize_on_meta", type=int, default=1, help="Whether to initialize parameters on meta device.", choices=[0, 1]
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    group.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default='fp16', help="Mixed precision option.", choices=['fp32', 'fp16', 'bf16'],
    )
    parser.add_argument(
        "--hetero_config", type=int, nargs='+', default=0, help="Give and execute heterogeneous configuration",
    )
    
    parser.add_argument(
        "--pp_layouts", type=str, default=None, help="Give rank layouts",
    )
    parser.add_argument(
        "--hetero_configs", type=str, default=None, help="Give pipeline layouts in general",
    )
    parser.add_argument(
        "--layer_partitions", type=str, default=None, help="Give layer partition layouts",
    ) 
    parser.add_argument(
        "--default_dp_type", type=str, default=None, help="Default data parallel type", choices=["ddp","zero2","zero3"],
    )
    parser.add_argument(
        "--pp_partition", type=int, nargs='+', default=0, help="Give and execute pipeline configuration",
    )
    parser.add_argument(
        "--total-layer-num", type=int, default=0, help='Total transformer layers to be trained',
    )
    parser.add_argument(
        "--checkpoint-layers", action='store_true', help='Whether apply activation recompute on each transformer layer'
    )
    parser.add_argument(
        "--checkpoint-all", action='store_true', help='Whether apply activation recompute on embedding, prenorm, cls layers'
    )
    parser.add_argument(
        "--seq-parallel", action='store_true', help='Whether apply sequence parallel'
    )
    parser.add_argument(
        "--accum-iter", type=int, default=1, help='Gradient accumulation cycles', 
    )

    # utils arguments
    parser.add_argument(
        "--token", type=str, default='', help="Access token to gated models",
    )

    # training arguments
    parser.add_argument(
        "--global_bsz_size", type=int, nargs='+', default=2, help="global_bsz_size",
    )
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=0, help="Each pipeline chunk num",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Training epochs",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate",
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--fsdp", type=int, default=1, help="Apply FSDP", choices=[0, 1],
    )
    parser.add_argument(
        "--apply_strategy", type=int, default=0, help="Apply searched strategy.", choices=[0, 1],
    )

    parser.add_argument('--profile', action='store_true', help='Enable time profiling')
    parser.add_argument('--profile-mem', action='store_true', help='Enable memory profiling')
    parser.add_argument('--run-iter', type=int, default=20)
    parser.add_argument('--load-params', action='store_true')
    parser.add_argument('--optimizer-type', type=str, default='adam')
    parser.add_argument('--display_one_pipeline', action='store_true')
    parser.add_argument('--all_hetero_configs_path', type=str, default=None, help="JSON file path to hetero configs")
    parser.add_argument('--pipeline-type', type=str, default="Gpipe", help="JSON file path to hetero configs")
    parser.add_argument('--recompute-stage-output', action='store_true', help="Whether to store stage output for backward")    
    
    return parser


_HETERO_GROUPS = None
def get_hetero_groups():
    global _HETERO_GROUPS
    return _HETERO_GROUPS

def set_hetero_groups(hetero_groups):
    global _HETERO_GROUPS
    _HETERO_GROUPS = hetero_groups
