export HTTPS_PROXY="http://cityu:upside_tumbling_turbine@klb-fwproxy-vip.aisc.local:3128"
export HTTP_PROXY="http://cityu:upside_tumbling_turbine@klb-fwproxy-vip.aisc.local:3128"
export https_proxy="http://cityu:upside_tumbling_turbine@klb-fwproxy-vip.aisc.local:3128"
export http_proxy="http://cityu:upside_tumbling_turbine@klb-fwproxy-vip.aisc.local:3128"
export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_IGNORE_GIT=True
export WANDB_INSECURE_DISABLE_SSL=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --master_port=29505 --nnodes=1 --nproc_per_node=1 pipe.py --config-name='v2_folding_sft.yaml'