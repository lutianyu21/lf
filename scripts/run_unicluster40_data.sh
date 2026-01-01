#!/bin/bash
# Unicluster40 数据集处理脚本
# 同时生成 structure (s/) 和 folding (p2s/) 类型的 tokenized parquet 文件

set -e

PROJECT_ROOT=/SPXvePFS/users/jtfeng/lf
cd ${PROJECT_ROOT}

# ========== 配置 ==========
PYTHON="/SPXvePFS/share/miniconda3/envs/lf/bin/python"

# Checkpoint 路径
TOKENIZER_CKPT="/SPXvePFS/share/zzhang/ckpt/v4-ar-epoch=00-val_loss=0.1949.ckpt"
STRUCTURE_CKPT="/SPXvePFS/share/zzhang/ckpt/v3-structure-epoch=04-val_rmsd=0.3359.ckpt"

# 数据路径 (可通过环境变量覆盖)
BQ_PATH="${BQ_PATH:-/SPXvePFS/share/llmfolding/lf/dataset/v3.1/bq_unicluster40.parquet}"
STRUCT_DIR="${STRUCT_DIR:-/SPXvePFS/share/llmfolding/lf/data/unicluster40/raw}"
DATASET="${DATASET:-unicluster40}"

# 输出目录
OUTPUT_DIR="/SPXvePFS/share/llmfolding/lf/dataset/v3.1"

#-------------------------------------------------------------------------------
# 硬件配置 (自动检测 GPU 数量)
#-------------------------------------------------------------------------------
# 优先使用 MLP 平台环境变量，否则使用 nvidia-smi 检测
if [ -n "$MLP_WORKER_GPU" ]; then
    NUM_GPUS=$MLP_WORKER_GPU
elif command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi

# 处理参数 (根据 GPU 数量自适应)
NUM_CONSUMERS=${NUM_CONSUMERS:-$NUM_GPUS}
NUM_PRODUCERS=${NUM_PRODUCERS:-$((NUM_CONSUMERS * 2))}
BSZ=${BSZ:-50}

# Ray 配置 (禁用不必要的功能，避免集群环境问题)
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=0

# ========== 函数 ==========
run_dataset() {
    local split_type=$1     # s (structure) 或 p2s (folding)
    local dataset_name="${split_type}/${DATASET}"

    echo "=========================================="
    echo "处理: ${DATASET} (split: ${dataset_name})"
    echo "=========================================="

    # 清理之前的临时文件
    rm -rf "${OUTPUT_DIR}/tmp"/*

    $PYTHON "${PROJECT_ROOT}/data_engine_runner.py" \
        --bq_path "${BQ_PATH}" \
        --parquet_dir "${OUTPUT_DIR}" \
        --tokenizer_ckpt "${TOKENIZER_CKPT}" \
        --structure_ckpt "${STRUCTURE_CKPT}" \
        --structure_dir "${STRUCT_DIR}" \
        --dataset_name "${dataset_name}" \
        --num_consumers ${NUM_CONSUMERS} \
        --num_producers ${NUM_PRODUCERS} \
        --bsz ${BSZ} \
        --ops merge shuffle split

    echo "[完成] ${dataset_name}"
    echo ""
}

# ========== 主流程 ==========
echo "=========================================="
echo "Unicluster40 数据集处理脚本"
echo "=========================================="
echo "数据集: ${DATASET}"
echo "BQ 文件: ${BQ_PATH}"
echo "结构目录: ${STRUCT_DIR}"
echo "Tokenizer: ${TOKENIZER_CKPT}"
echo "Structure: ${STRUCTURE_CKPT}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "=== 硬件配置 ==="
echo "检测到 GPU 数量: ${NUM_GPUS}"
echo "NUM_CONSUMERS: ${NUM_CONSUMERS}"
echo "NUM_PRODUCERS: ${NUM_PRODUCERS}"
echo "BSZ: ${BSZ}"
echo ""

# 停止之前的 ray 进程
ray stop --force 2>/dev/null || true
sleep 2

# 处理 structure (s/unicluster40)
echo ">>> 处理 structure 数据 (s/${DATASET})..."
run_dataset "s"

# 处理 folding (p2s/unicluster40)
echo ">>> 处理 folding 数据 (p2s/${DATASET})..."
run_dataset "p2s"

# 清理 ray
ray stop --force 2>/dev/null || true

echo "=========================================="
echo "全部完成!"
echo ""
echo "输出目录:"
echo "  structure: ${OUTPUT_DIR}/structure/${DATASET}/"
ls -la "${OUTPUT_DIR}/structure/${DATASET}/" 2>/dev/null || echo "  (目录不存在)"
echo ""
echo "  folding: ${OUTPUT_DIR}/folding/${DATASET}/"
ls -la "${OUTPUT_DIR}/folding/${DATASET}/" 2>/dev/null || echo "  (目录不存在)"
echo "=========================================="
