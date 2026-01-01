#!/bin/bash
# Debug training script for LLMFolding on Volces cluster
# Uses small debug dataset for quick iteration

# Disable WandB for debugging
export WANDB_MODE=disabled

# Cache configuration
export HF_DATASETS_CACHE=.cache/hf_datasets

# CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Volces cluster paths
export LF_CHECKPOINT_ROOT=/SPXvePFS/share/jiangtao/checkpoints/LLMFolding
export LF_DATA_ROOT=/SPXvePFS/share/zzhang/LLMFolding_tokenizer/debug_data_ar

# Training configuration (default: single GPU for debugging)
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29505}

echo "=== Debug Training ==="
echo "Node: $(hostname)"
echo "GPUs: $NPROC_PER_NODE"
echo "Checkpoint root: $LF_CHECKPOINT_ROOT"
echo "Data root: $LF_DATA_ROOT"

echo "=== GPU visibility ==="
nvidia-smi -L

echo "=== Running debug task ==="
torchrun \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    pipe.py --config-name='debug'

echo "=== Debug training completed ==="
