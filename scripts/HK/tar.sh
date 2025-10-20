#!/bin/bash

# ====================================================================
# 脚本功能: 批量解压指定目录下所有非 .partial 的 tar 压缩包
#           已存在的文件将直接跳过，保留进度显示
# ====================================================================

# 1. 定义源目录和目标目录变量
SOURCE_DIR="/home/projects/protein/lutianyu/data/AFDB/cityu-data/tsinghua-ai4science/part_00"
DEST_DIR="/home/projects/protein/lutianyu/data/AFDB/AF_part00"

# 2. 确保目标目录存在 (如果不存在则创建)
mkdir -p "$DEST_DIR"

echo "========================================="
echo "开始批量解压 AlphaFold 数据"
echo "源目录: $SOURCE_DIR"
echo "目标目录: $DEST_DIR"
echo "-----------------------------------------"

# 3. 统计总文件数（非 .partial 的 tar 文件）
TOTAL=$(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.tar" ! -name "*.partial*" | wc -l)
COUNT=0

# 4. 循环遍历源目录下的 tar 文件
for ARCHIVE in "$SOURCE_DIR"/*.tar; do

    # 跳过不存在的文件（防止通配符没有匹配时报错）
    [ -e "$ARCHIVE" ] || continue

    FILENAME=$(basename "$ARCHIVE")

    # 排除 .partial 文件
    if [[ "$FILENAME" == *".partial"* ]]; then
        echo "--> [跳过] 发现 .partial 文件: $FILENAME"
        continue
    fi

    # 增加计数
    COUNT=$((COUNT + 1))
    echo "--> [$COUNT/$TOTAL] 正在处理: $FILENAME"

    # 使用 tar 解压，直接跳过已存在文件
    tar -xf "$ARCHIVE" -C "$DEST_DIR" --skip-old-files

    if [ $? -eq 0 ]; then
        echo "    [成功] 解压完成。"
    else
        echo "    [失败] 解压 $FILENAME 时出错！"
    fi
done

echo "-----------------------------------------"
echo "所有文件批量解压操作完成。"
echo "========================================="