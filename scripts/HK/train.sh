export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
torchrun --nproc_per_node=1 trainer_new.py --config-name='folding'
