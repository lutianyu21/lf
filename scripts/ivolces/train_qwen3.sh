export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_BASE_URL=https://api.bandw.top
torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    trainer_qwen3.py \
    --config-name='folding'
    
