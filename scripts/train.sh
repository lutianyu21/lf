export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_BASE_URL=https://api.bandw.top
torchrun --nproc_per_node=1 trainer.py --config-name='folding'