#!/bin/bash
#===============================================================================
# LLMFolding Training Script for Volces Cluster
#===============================================================================

#-------------------------------------------------------------------------------
# 1. TRAINING PARAMETERS (modify these for different experiments)
#-------------------------------------------------------------------------------
# Experiment configuration
CONFIG_NAME="pretrain"                # Self-contained config template
# CONFIG_NAME="v3-1_pt_volces"        # Use this for volces-specific config
# CONFIG_NAME="debug"                 # Use this for quick testing

# Run name (used for checkpoint dir, wandb name, and config name)
RUN_NAME="debug"

# Data paths (input)
TRAIN_DATA="/SPXvePFS/share/jiangtao/data/LLMFolding/v4-rmsd=2.3/parquet/train.parquet"
# Evaluation datasets
EVAL_CAMEO="/SPXvePFS/share/llmfolding/lf/dataset/dataset_dist3-1/folding/benchmark/cameo2022.parquet"
EVAL_CASP15="/SPXvePFS/share/llmfolding/lf/dataset/dataset_dist3-1/folding/benchmark/casp15.parquet"
EVAL_CASP16="/SPXvePFS/share/llmfolding/lf/dataset/dataset_dist3-1/folding/benchmark/casp16.parquet"
# Hydra list format: [path1,path2,path3]
EVAL_DATA="${EVAL_CAMEO},${EVAL_CASP15},${EVAL_CASP16}"

# Output directory
OUTPUT_ROOT="/SPXvePFS/share/jiangtao/checkpoints/LLMFolding"
CHECKPOINT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# Batch size configuration
# Global batch size = num_gpus × gradient_accumulation_steps × per_device_batch_size
GLOBAL_BATCH_SIZE=256                 # Target global batch size
PER_DEVICE_BATCH_SIZE=8               # Batch size per GPU

# Training overrides (uncomment to override config defaults)
# TRAIN_OVERRIDES="trainer.max_steps=100000"
# TRAIN_OVERRIDES="$TRAIN_OVERRIDES trainer.learning_rate=5e-4"

#-------------------------------------------------------------------------------
# 2. GLOBAL PATH CONFIGURATION (Volces cluster paths)
#-------------------------------------------------------------------------------
# Project root
export LF_ROOT=/SPXvePFS/users/jtfeng/lf
cd ${LF_ROOT}

# Data paths
export LF_DATA_ROOT=/SPXvePFS/share/llmfolding/lf/data
export LF_MODEL_ROOT=/SPXvePFS/model
export LF_TOKENIZER_CKPT_ROOT=/SPXvePFS/share/zzhang/LLMFolding_tokenizer/ckpt

# Tokenizer/Structure checkpoints (optional, uses defaults if not set)
# Available tokenizers: v4-epoch=46-val_loss=0.1712.ckpt (default), v4-epoch=00, v4-ar-epoch=00
export LF_TOKENIZER_CKPT=/SPXvePFS/share/zzhang/ckpt/v4-epoch=46-val_loss=0.1712.ckpt
export LF_STRUCTURE_CKPT=/SPXvePFS/share/zzhang/ckpt/v3-structure-epoch=04-val_rmsd=0.3359.ckpt

# Output paths (use OUTPUT_ROOT from training parameters)
export LF_CHECKPOINT_ROOT=${OUTPUT_ROOT}

#-------------------------------------------------------------------------------
# 3. HARDWARE CONFIGURATION
#-------------------------------------------------------------------------------
# Conda environment
CONDA_ENV="/SPXvePFS/share/miniconda3/envs/lf"
TORCHRUN="$CONDA_ENV/bin/torchrun"

# Distributed training (MLP platform environment variables)
# MLP_WORKER_NUM: number of nodes, MLP_WORKER_GPU: GPUs per node
# MLP_ROLE_INDEX: node rank, MLP_WORKER_0_HOST/PORT: master addr/port
NNODES=${MLP_WORKER_NUM:-${NNODES:-1}}
NPROC_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-localhost}
MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-29505}}

# Auto-calculate gradient accumulation steps
# Formula: GLOBAL_BATCH_SIZE = num_gpus × gradient_accumulation_steps × per_device_batch_size
NUM_GPUS=$((NNODES * NPROC_PER_NODE))
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_DEVICE_BATCH_SIZE)))
# Ensure at least 1
if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=1
fi

# CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#-------------------------------------------------------------------------------
# 4. ENVIRONMENT CONFIGURATION
#-------------------------------------------------------------------------------
# WandB configuration
# export WANDB_BASE_URL=https://api.bandw.top     # Uncomment to use custom server (app.bandw.top)
export WANDB_ENTITY="LLMFolding"                    # Team name on wandb.ai
export WANDB_PROJECT="LF-jt"
export WANDB_NAME="${RUN_NAME}"                     # Inherited from RUN_NAME
export WANDB_DIR=${OUTPUT_ROOT}                     # WandB logs under OUTPUT_ROOT
export WANDB_IGNORE_GIT=True
# export WANDB_MODE=offline                        # Uncomment for offline mode

# Cache
export HF_DATASETS_CACHE=.cache/hf_datasets

#===============================================================================
# LAUNCH SCRIPT (usually no need to modify below)
#===============================================================================
cd "$LF_ROOT"

# Create output directory if it doesn't exist
mkdir -p ${CHECKPOINT_DIR}

echo "============================================================"
echo "LLMFolding Training - $(date)"
echo "============================================================"
if [ -n "$WANDB_BASE_URL" ]; then
    echo "WandB URL: https://app.bandw.top/${WANDB_ENTITY}/${WANDB_PROJECT}"
else
    echo "WandB URL: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
fi

echo ""
echo "=== Node Information ==="
hostname

echo ""
echo "=== GPU Visibility ==="
nvidia-smi -L

echo ""
echo "=== Data Configuration ==="
echo "RUN_NAME:             $RUN_NAME"
echo "TRAIN_DATA:           $TRAIN_DATA"
echo "EVAL_DATA:            $EVAL_DATA"
echo "CHECKPOINT_DIR:       $CHECKPOINT_DIR"

echo ""
echo "=== Path Configuration ==="
echo "LF_ROOT:              $LF_ROOT"
echo "LF_MODEL_ROOT:        $LF_MODEL_ROOT"
echo "LF_TOKENIZER_CKPT_ROOT: $LF_TOKENIZER_CKPT_ROOT"

echo ""
echo "=== Training Configuration ==="
echo "CONFIG_NAME:          $CONFIG_NAME"
echo "NNODES:               $NNODES"
echo "NPROC_PER_NODE:       $NPROC_PER_NODE"
echo "NODE_RANK:            $NODE_RANK"
echo "MASTER_ADDR:          $MASTER_ADDR"
echo "MASTER_PORT:          $MASTER_PORT"

echo ""
echo "=== Batch Size Configuration ==="
echo "GLOBAL_BATCH_SIZE:    $GLOBAL_BATCH_SIZE"
echo "PER_DEVICE_BATCH_SIZE: $PER_DEVICE_BATCH_SIZE"
echo "NUM_GPUS:             $NUM_GPUS"
echo "GRADIENT_ACCUM_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS * PER_DEVICE_BATCH_SIZE))"

echo ""
echo "=== Starting Training ==="
$TORCHRUN \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pipe.py --config-name="$CONFIG_NAME" \
    "name=${RUN_NAME}" \
    "dataset.train=[${TRAIN_DATA}]" \
    "dataset.eval=[${EVAL_DATA}]" \
    "trainer.output_dir=${CHECKPOINT_DIR}" \
    "trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
    "trainer.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE}" \
    "hydra.run.dir=${OUTPUT_ROOT}" \
    $TRAIN_OVERRIDES

echo ""
echo "=== Training Completed - $(date) ==="
