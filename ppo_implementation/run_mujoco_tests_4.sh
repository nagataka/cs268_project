#!/usr/bin/env bash

for run in {1..3}
do
    OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
    OPENAI_LOGDIR=tests/half_cheetah_PPO_adapt_penalty_1_env_run_${run}_$(date '+%d-%m-%Y_%H:%M:%S') \
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
    --beta 1 \
    --learning_rate 3e-4 \
    --max_grad_norm 0.5 \
    --kl_regularization_method 'adaptive-penalty' \
    --clip_ratio 0.2 \
    --entropy_coef 0. \
    --vf_coef 1. \
    --joint_network 'False' \
    --seed $run \
    --save_best_model 'True' \
    --min_save_interval_seconds 120 \
    --restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.
done