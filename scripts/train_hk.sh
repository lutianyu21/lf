export WANDB_IGNORE_GIT=True
export WANDB_INSECURE_DISABLE_SSL=True
export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
torchrun --nproc_per_node=1 trainer.py --config-name='folding'