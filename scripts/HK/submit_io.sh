#!/bin/bash
#SBATCH --account=protein
#SBATCH --partition=AISS2024110101
#SBATCH --job-name=lf-ray
#SBATCH --output=output_ray.log
#SBATCH --error=error_ray.err
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=2000
#SBATCH --export=ALL

CONTAINER_PATH=/home/projects/protein/lutianyu/images/modern.sqsh
CONTAINER_NAME=modern

# -------------------------------
# 环境变量配置
# -------------------------------
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=NVL
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=28
export TORCH_DISTRIBUTED_TIMEOUT=1800

# 网络配置
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
port=6379

echo "=== Ray Cluster Start ==="
echo "Head node: $head_node ($head_ip)"
echo "Nodes: ${nodes[@]}"
echo "========================="

# -------------------------------
# Step 1. 在所有节点上确保容器存在
# -------------------------------
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=1 bash -c "
if ! enroot list | grep -q '^$CONTAINER_NAME\$'; then
    echo \"[ \$(hostname) ] Creating container: $CONTAINER_NAME\"
    enroot create --name $CONTAINER_NAME $CONTAINER_PATH
else
    echo \"[ \$(hostname) ] Container already exists.\"
fi
"

# -------------------------------
# Step 2. 启动 Ray Head 节点
# -------------------------------
echo "=== Starting Ray head on $head_node ==="
srun --nodes=1 --ntasks=1 -w $head_node enroot start -r \
    --mount /home/projects/protein/lutianyu:/GenSIvePFS/users/lutianyu \
    --mount /home/projects/protein/zhangzhe/protenix_data/mmcif:/GenSIvePFS/users/lutianyu/lf/data/rcsb_mmcif \
    -w $CONTAINER_NAME \
    -- bash -c "
        ray stop >/dev/null 2>&1;
        ray start --head --node-ip-address=$head_ip --port=$port --num-cpus=224 --num-gpus=8;
        echo 'Ray head started on $head_ip:$port';
        sleep infinity;
    " &

sleep 20  # 等待 head 稳定启动

# -------------------------------
# Step 3. 启动 Ray Worker 节点
# -------------------------------
for node in "${nodes[@]:1]}"; do
    echo "=== Starting Ray worker on $node ==="
    srun --nodes=1 --ntasks=1 -w $node enroot start -r \
        --mount /home/projects/protein/lutianyu:/GenSIvePFS/users/lutianyu \
        --mount /home/projects/protein/zhangzhe/protenix_data/mmcif:/GenSIvePFS/users/lutianyu/lf/data/rcsb_mmcif \
        -w $CONTAINER_NAME \
        -- bash -c "
            ray stop >/dev/null 2>&1;
            ray start --address='$head_ip:$port' --num-cpus=224 --num-gpus=8;
            echo 'Worker joined $head_ip:$port';
            sleep infinity;
        " &
done

sleep 30  # 等所有 worker 注册完毕

# -------------------------------
# Step 4. 提交 Ray Python 脚本（只在 Head 节点执行）
# -------------------------------
echo "=== Running Ray task on head ==="
srun --nodes=1 --ntasks=1 -w $head_node enroot start -r \
    -w $CONTAINER_NAME \
    -- bash -c "
        export RAY_DEDUP_LOGS=0;
        export TMPDIR=/GenSIvePFS/users/lutianyu/lf/tmp;
        mkdir -p \$TMPDIR;
        cd /GenSIvePFS/users/lutianyu/lf;
        conda run -n qwen3 python t.py;
        echo '=== Ray job done ===';
        ray stop;
    "

# -------------------------------
# Step 5. 清理集群
# -------------------------------
echo "=== Cleaning up Ray workers ==="
for node in "${nodes[@]}"; do
    srun --nodes=1 --ntasks=1 -w $node enroot start -r -w $CONTAINER_NAME \
        -- bash -c "ray stop" &
done

wait
echo "=== All Ray processes stopped ==="