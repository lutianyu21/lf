#!/bin/bash
#===============================================================================
# Text + Structure Tokenization Script
# Processes text_with_struct_id.parquet to add structure tokens
#===============================================================================

# Project root
export LF_ROOT=/SPXvePFS/users/jtfeng/lf
cd ${LF_ROOT}

#-------------------------------------------------------------------------------
# 1. DATA CONFIGURATION
#-------------------------------------------------------------------------------
# Input/Output paths
INPUT_PARQUET="/SPXvePFS/users/jtfeng/lf/data/text_with_struct_id.parquet"
OUTPUT_DIR="/SPXvePFS/users/jtfeng/lf/data/text_struct_tokenized"

# Structure data directories
AFDB_DIR="/SPXvePFS/share/zzhang/AFDB"
PDB_DIR="/SPXvePFS/share/zzhang/PDB/mmcif"

# Tokenizer checkpoints
TOKENIZER_CKPT="/SPXvePFS/share/zzhang/LLMFolding_tokenizer/ckpt/v4-epoch=46-val_loss=0.1712.ckpt"
STRUCTURE_CKPT="/SPXvePFS/share/zzhang/LLMFolding_tokenizer/ckpt/v3-structure-epoch=04-val_rmsd=0.3359.ckpt"

#-------------------------------------------------------------------------------
# 2. PROCESSING PARAMETERS
#-------------------------------------------------------------------------------
# Batch size per GPU worker
BATCH_SIZE=32

# Number of data producers (parallel parquet readers)
NUM_PRODUCERS=2

# Whether to merge shards at the end
MERGE_SHARDS="--merge-shards"
# MERGE_SHARDS=""  # Uncomment to skip merging

#-------------------------------------------------------------------------------
# 3. HARDWARE CONFIGURATION
#-------------------------------------------------------------------------------
# Conda environment
CONDA_ENV="/SPXvePFS/share/miniconda3/envs/lf"
PYTHON="$CONDA_ENV/bin/python"

# GPU configuration (MLP platform environment variables)
# MLP_WORKER_GPU: GPUs per node from platform
NPROC_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-4}}
NUM_GPUS=${NPROC_PER_NODE}

# CUDA settings
export CUDA_HOME=/usr/local/cuda-12.4
export PYTHONPATH=/SPXvePFS/share/zzhang/LLMFolding_tokenizer:$PYTHONPATH

#===============================================================================
# LAUNCH SCRIPT (usually no need to modify below)
#===============================================================================
cd "$LF_ROOT"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "Text + Structure Tokenization - $(date)"
echo "============================================================"

echo ""
echo "=== Node Information ==="
hostname

echo ""
echo "=== GPU Visibility ==="
nvidia-smi -L

echo ""
echo "=== Data Configuration ==="
echo "INPUT_PARQUET:        $INPUT_PARQUET"
echo "OUTPUT_DIR:           $OUTPUT_DIR"
echo "AFDB_DIR:             $AFDB_DIR"
echo "PDB_DIR:              $PDB_DIR"

echo ""
echo "=== Processing Configuration ==="
echo "NUM_GPUS:             $NUM_GPUS"
echo "BATCH_SIZE:           $BATCH_SIZE"
echo "NUM_PRODUCERS:        $NUM_PRODUCERS"

echo ""
echo "=== Starting Processing ==="
$PYTHON \
    ${LF_ROOT}/scripts/data_processing/text_struct_tokenizer.py \
    --input "$INPUT_PARQUET" \
    --output-dir "$OUTPUT_DIR" \
    --afdb-dir "$AFDB_DIR" \
    --pdb-dir "$PDB_DIR" \
    --tokenizer-ckpt "$TOKENIZER_CKPT" \
    --structure-ckpt "$STRUCTURE_CKPT" \
    --batch-size $BATCH_SIZE \
    --num-gpus $NUM_GPUS \
    --num-producers $NUM_PRODUCERS \
    $MERGE_SHARDS \
    "$@"

echo ""
echo "=== Processing Completed - $(date) ==="
echo "Output directory: $OUTPUT_DIR"
