# export NCCL_P2P_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_ALGO=Ring
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=9996
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 torchrun $DISTRIBUTED_ARGS llama_train.py \
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun $DISTRIBUTED_ARGS llama_train.py \
torchrun $DISTRIBUTED_ARGS llama_train.py \
--model_size llama-7b \
--mixed_precision fp16 \
--use-flash-attn \
--total-layer-num 16 \
--hetero_configs "[[1] * 2] * 4" \
--layer_partitions "[[8, 8] ] * 4" \
--chunks 4  \
--global_bsz_size 4 \
--accum-iter 1 \
--run-iter 2 \
--fp16 \
--display_one_pipeline \
--checkpoint-layers \
--checkpoint-all \
# --recompute-stage-output \
# --default_dp_type ddp \
# --pp_layouts "[[[0, 1, 4, 6], [2, 5, 3, 7]]]" \