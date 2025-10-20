#!/bin/bash

# ====================================================================
# 脚本功能: 批量解压指定目录下所有非 .partial 的 tar 压缩包
#           如果目标目录已存在同名文件，跳过整个 tar 包
#           显示进度
# ====================================================================

SOURCE_DIR="/home/projects/protein/lutianyu/data/AFDB/cityu-data/tsinghua-ai4science/part_00"
DEST_DIR="/home/projects/protein/lutianyu/data/AFDB/AF_part00"

mkdir -p "$DEST_DIR"

echo "========================================="
echo "开始批量解压 AlphaFold 数据"
echo "源目录: $SOURCE_DIR"
echo "目标目录: $DEST_DIR"
echo "-----------------------------------------"

# 获取所有 tar 文件总数
ALL_ARCHIVES=("$SOURCE_DIR"/*.tar)
TOTAL=${#ALL_ARCHIVES[@]}
COUNT=0

for ARCHIVE in "$SOURCE_DIR"/*.tar; do
    [ -f "$ARCHIVE" ] || continue

    FILENAME=$(basename "$ARCHIVE")

    # 跳过 .partial 文件
    if [[ "$FILENAME" == *".partial"* ]]; then
        echo "--> [跳过] 发现 .partial 文件: $FILENAME"
        continue
    fi

    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] 正在处理: $FILENAME"

    # 检查 tar 内文件是否全部已经存在目标目录
    ALL_EXIST=true
    while IFS= read -r FILE; do
        if [ ! -f "$DEST_DIR/$FILE" ]; then
            ALL_EXIST=false
            break
        fi
    done < <(tar -tf "$ARCHIVE")

    if [ "$ALL_EXIST" = true ]; then
        echo "    [跳过] 所有文件已存在，跳过整个 tar 包。"
        continue
    fi

    # 解压 tar 包到目标目录
    tar -xf "$ARCHIVE" -C "$DEST_DIR"
    if [ $? -eq 0 ]; then
        echo "    [成功] 解压完成。"
    else
        echo "    [失败] 解压 $FILENAME 时出错！"
    fi
done

echo "-----------------------------------------"
echo "所有文件批量解压操作完成。"
echo "========================================="