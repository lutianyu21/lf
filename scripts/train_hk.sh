echo "=== Node Information ==="
hostname

echo "=== GPU visibility ==="
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export HTTPS_PROXY="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export HTTP_PROXY="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export https_proxy="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export http_proxy="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_IGNORE_GIT=True
export WANDB_INSECURE_DISABLE_SSL=True

echo "=== Changing pipe host ==="
pip config list
pip config set global.index-url https://pypi.org/simple
pip config list

echo "=== Pip installing  ==="
PIP=/root/miniconda3/envs/dplm/bin/pip
$PIP show torch >/dev/null 2>&1 || $PIP install \
    --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

$PIP show ByProt >/dev/null 2>&1 || $PIP install \
    --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
    -e /AIRvePFS/ai4science/users/tianyu/lf/utils/dplm_utils/dplm

$PIP show openfold >/dev/null 2>&1 || $PIP install \
    --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
    -e /AIRvePFS/ai4science/users/tianyu/lf/utils/dplm_utils/dplm/vendor/openfold


echo "=== Starting torchrun ==="
cd /AIRvePFS/ai4science/users/tianyu/lf/utils/dplm_utils/dplm
conda run -n dplm torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPU_COUNT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    trainer.py --config-name='folding'

echo "=== torchrun command executed ==="