python RL/train_combustion_ppo.py \
    --track True \
    --wandb_project_name "combustion_control" \
    --total_timesteps 100000 \
    --num_envs 4 \
    --learning_rate 2.5e-4 \
    --use_temporal_features True \
    --use_species_features True \
    --use_stiffness_features False