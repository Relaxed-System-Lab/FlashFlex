from initialize import initialize
from pymetis import Options
from dataclasses import dataclass
from arguments import get_args


@dataclass
class Config:
    args = get_args()

    model_size = args.model_size

    # graph partition config
    niter = args.niter
    options = Options(contig=True)
    npipeline = args.npipeline
    param = [2, 0.2]  # n, p for binomial
    K = initialize(param, (1, npipeline))

    # pipeline 
    kway = args.kway

    # network config
    inter_bw = args.inter_bw
    specs = None

    # utils
    device_machine_map = None

    # model config
    
    GLB_B = args.global_bsz * args.accum_iter
    # GLB_B = 5000
    GLB_MB = args.micro_bsz * args.accum_iter
    assert GLB_B % GLB_MB  == 0
    N_MB = GLB_B // GLB_MB
    assert GLB_MB >= npipeline, "Too many pipelines"
    B = GLB_B // npipeline
    MB = GLB_MB // npipeline
    if args.MB is not None:
        MB = args.MB


    S = 4096

    if model_size == 'llama-30b':
        H = 6656
        L = 60
        N_attn_heads = 52
        P = 30
    elif model_size == 'llama-7b':
        H = 4096
        L = 32
        N_attn_heads = 32
        P = 7
    elif model_size == 'llama-13b':
        H = 5120
        L = 40
        N_attn_heads = 40
        P = 13
    elif model_size == 'llama-70b':
        H = 8192
        L = 80
        N_attn_heads = 64
        P = 7

    V = 32000
    B_type = 2
    
    T = GLB_B * S