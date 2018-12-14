#!/usr/bin/env bash

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so \
python ppo.py \
    --env_id 'HalfCheetah-v2' \
    --num_workers 1 \
    --total_timesteps 1e6 \
    --num_train_updates_per_epoch 10 \
    --num_minibatches_per_epoch_train_update 32 \
    --time_horizon 2048 \
    --gamma 0.99 \
    --lam 0.95 \
    --target_kl 0.01 \
    --beta 3 \
    --learning_rate 3e-4 \
    --max_grad_norm 0.5 \
    --kl_regularization_method 'clip' \
    --clip_ratio 0.2 \
    --entropy_coef 0. \
    --vf_coef 1. \
    --joint_network 'False' \
    --seed 42 \
    --save_best_model 'True' \
    --min_save_interval_seconds 120 \
    --restore_from_checkpoint 'tests/half_cheetah_PPO_default_1_env_run_2_12-12-2018_15:57:39/saved_model/model.ckpt' \
    --do_demo 'True'