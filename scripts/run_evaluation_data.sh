#!/bin/bash
# 评估数据集处理脚本
# 用于生成 casp15/casp16/cameo2022 的 tokenized parquet 文件
# 支持两种任务类型: p2s (folding) 和 psps (cfolding)

set -e

PROJECT_ROOT=/SPXvePFS/users/jtfeng/lf
cd ${PROJECT_ROOT}

# ========== 配置 ==========
PYTHON="/SPXvePFS/share/miniconda3/envs/lf/bin/python"

# Checkpoint 路径
TOKENIZER_CKPT="/SPXvePFS/share/zzhang/ckpt/v4-ar-epoch=00-val_loss=0.1949.ckpt"
STRUCTURE_CKPT="/SPXvePFS/share/zzhang/ckpt/v3-structure-epoch=04-val_rmsd=0.3359.ckpt"

# 数据路径
BQ_DIR="/SPXvePFS/share/jiangtao/evaluation/bq"
STRUCTURE_DIR="/SPXvePFS/share/jiangtao/evaluation/structure"

# 输出目录
OUTPUT_DIR="/SPXvePFS/share/llmfolding/lf/dataset/v3.1"

# 处理参数
NUM_CONSUMERS=1
NUM_PRODUCERS=2
BSZ=10

# 任务类型: p2s (folding) 或 psps (cfolding)
TASK_TYPE="${TASK_TYPE:-all}"

# ========== 函数 ==========
run_benchmark() {
    local name=$1
    local bq_file=$2
    local struct_subdir=$3
    local task_type=$4

    local dataset_name="${task_type}/${name}"
    local output_subdir="folding"
    if [ "$task_type" == "psps" ]; then
        output_subdir="cfolding"
    fi

    echo "=========================================="
    echo "处理: ${name} (${task_type})"
    echo "=========================================="

    # 清理之前的临时文件
    rm -rf "${OUTPUT_DIR}/tmp"/*

    $PYTHON "${PROJECT_ROOT}/data_engine_runner.py" \
        --bq_path "${BQ_DIR}/${bq_file}" \
        --parquet_dir "${OUTPUT_DIR}" \
        --tokenizer_ckpt "${TOKENIZER_CKPT}" \
        --structure_ckpt "${STRUCTURE_CKPT}" \
        --structure_dir "${STRUCTURE_DIR}/${struct_subdir}" \
        --dataset_name "${dataset_name}" \
        --num_consumers ${NUM_CONSUMERS} \
        --num_producers ${NUM_PRODUCERS} \
        --bsz ${BSZ} \
        --ops merge

    echo "[完成] ${name} -> ${OUTPUT_DIR}/${output_subdir}/benchmark/"
    echo ""
}

show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --task p2s|psps|all   任务类型 (默认: p2s)"
    echo "                        p2s  = folding (蛋白质结构预测)"
    echo "                        psps = cfolding (条件折叠)"
    echo "                        all  = 同时处理 p2s 和 psps"
    echo "  --dataset NAME        处理单个数据集 (casp15|casp16|cameo2022|all)"
    echo "  --output DIR          输出目录 (默认: ${OUTPUT_DIR})"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "示例:"
    echo "  $0                           # 处理所有 p2s 数据集"
    echo "  $0 --task psps               # 处理所有 psps (cfolding) 数据集"
    echo "  $0 --task all                # 处理所有任务类型"
    echo "  $0 --dataset casp15          # 只处理 casp15"
    echo "  $0 --task psps --dataset casp15  # 处理 casp15 的 cfolding"
}

# ========== 解析参数 ==========
DATASET="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_TYPE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ========== 主流程 ==========
echo "=========================================="
echo "评估数据集处理脚本"
echo "=========================================="
echo "任务类型: ${TASK_TYPE}"
echo "数据集: ${DATASET}"
echo "Tokenizer: ${TOKENIZER_CKPT}"
echo "Structure: ${STRUCTURE_CKPT}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 停止之前的 ray 进程
ray stop --force 2>/dev/null || true
sleep 2

run_task() {
    local task=$1

    if [ "$DATASET" == "all" ] || [ "$DATASET" == "casp15" ]; then
        run_benchmark "casp15" "bq_casp15.parquet" "casp15" "$task"
    fi

    if [ "$DATASET" == "all" ] || [ "$DATASET" == "casp16" ]; then
        run_benchmark "casp16" "bq_casp16.parquet" "casp16" "$task"
    fi

    if [ "$DATASET" == "all" ] || [ "$DATASET" == "cameo2022" ]; then
        run_benchmark "cameo2022" "bq_cameo.parquet" "cameo2022" "$task"
    fi
}

# 根据任务类型执行
if [ "$TASK_TYPE" == "all" ]; then
    echo ">>> 处理 p2s (folding) 任务..."
    run_task "p2s"
    echo ">>> 处理 psps (cfolding) 任务..."
    run_task "psps"
elif [ "$TASK_TYPE" == "p2s" ] || [ "$TASK_TYPE" == "psps" ]; then
    run_task "$TASK_TYPE"
else
    echo "错误: 未知任务类型 '${TASK_TYPE}'"
    show_usage
    exit 1
fi

# 清理 ray
ray stop --force 2>/dev/null || true

echo "=========================================="
echo "全部完成!"
echo ""
echo "输出目录:"
if [ "$TASK_TYPE" == "all" ] || [ "$TASK_TYPE" == "p2s" ]; then
    echo "  folding: ${OUTPUT_DIR}/folding/benchmark/"
    ls -la "${OUTPUT_DIR}/folding/benchmark/" 2>/dev/null || echo "  (目录为空)"
fi
if [ "$TASK_TYPE" == "all" ] || [ "$TASK_TYPE" == "psps" ]; then
    echo "  cfolding: ${OUTPUT_DIR}/cfolding/benchmark/"
    ls -la "${OUTPUT_DIR}/cfolding/benchmark/" 2>/dev/null || echo "  (目录为空)"
fi
echo "=========================================="
