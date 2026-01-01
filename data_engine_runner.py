#!/usr/bin/env python
"""
Data Engine Runner - 命令行工具，用于处理蛋白质数据并生成 parquet 文件

Usage:
    python data_engine_runner.py \
        --bq_path /path/to/bq.parquet \
        --parquet_dir /path/to/output \
        --tokenizer_ckpt /path/to/discrete_tokenizer.ckpt \
        --structure_ckpt /path/to/structure_head.ckpt \
        --dataset_name p2s/unicluster40
"""

import argparse
from pathlib import Path

import ray


def main():
    parser = argparse.ArgumentParser(
        description="Data Engine Runner - 处理蛋白质数据并生成 parquet 文件"
    )

    # 必需参数
    parser.add_argument(
        "--bq_path",
        type=str,
        required=True,
        help="Bigtable parquet 文件路径"
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        required=True,
        help="输出 parquet 文件的目录"
    )
    parser.add_argument(
        "--tokenizer_ckpt",
        type=str,
        required=True,
        help="Discrete tokenizer checkpoint 路径 (e.g., v4-epoch=46-val_loss=0.1712.ckpt)"
    )
    parser.add_argument(
        "--structure_ckpt",
        type=str,
        required=True,
        help="Structure head checkpoint 路径 (e.g., v3-structure-epoch=04-val_rmsd=0.3359.ckpt)"
    )
    parser.add_argument(
        "--qwen_tokenizer_path",
        type=str,
        default="/SPXvePFS/model/Qwen3-0.6B",
        help="Qwen tokenizer 路径 (default: /SPXvePFS/model/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        required=True,
        help="蛋白质结构文件目录 (包含 .cif/.pdb 文件)"
    )

    # 可选参数
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="dist3",
        choices=["dplm", "dist2", "dist3"],
        help="Tokenizer 类型 (default: dist3)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="p2s/unicluster40",
        help="数据集名称 (default: p2s/unicluster40)"
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=100,
        help="Batch size (default: 100)"
    )
    parser.add_argument(
        "--num_consumers",
        type=int,
        default=8,
        help="GPU worker 数量 (default: 8)"
    )
    parser.add_argument(
        "--num_producers",
        type=int,
        default=10,
        help="Producer worker 数量 (default: 10)"
    )
    parser.add_argument(
        "--ops",
        type=str,
        nargs="+",
        default=["merge", "shuffle", "split"],
        help="操作列表 (default: merge shuffle split)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="最大序列长度，超过的蛋白将被跳过 (default: 4096)"
    )

    args = parser.parse_args()

    # 初始化 Ray (禁用 metrics 避免集群环境下的连接问题)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _metrics_export_port=None,
    )

    # 导入 DataEngine（在 ray.init 之后）
    from utils.lf_utils.data_engine import DataEngine

    # 运行数据处理流程
    DataEngine.pipe(
        dataset_dir=Path(args.parquet_dir),
        bq_path=Path(args.bq_path),
        structure_dir=Path(args.structure_dir),
        bsz=args.bsz,
        num_consumers=args.num_consumers,
        num_producers=args.num_producers,
        tokenizer_name=args.tokenizer_name,
        tokenizer_ckpt=args.tokenizer_ckpt,
        structure_ckpt=args.structure_ckpt,
        qwen_tokenizer_path=args.qwen_tokenizer_path,
        dataset_name=args.dataset_name,
        ops=args.ops,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
