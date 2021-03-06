#!/usr/bin/env bash

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_A2C_8_envs_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=a2c --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_acer_8_envs_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=acer --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7


worker_amounts=(4 8 16 32)
for num_w in "${worker_amounts[@]}"
do
    OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
    OPENAI_LOGDIR=tests/breakout_ppo_${num_w}_workers_$(date '+%d-%m-%Y_%H:%M:%S') \
    python ppo.py \
    --env_id 'BreakoutNoFrameskip-v4' \
    --num_workers $num_w \
    --total_timesteps 1e7 \
    --num_train_updates_per_epoch 4 \
    --num_minibatches_per_epoch_train_update 4 \
    --time_horizon 128 \
    --gamma 0.99 \
    --lam 0.95 \
    --target_kl 0.01 \
    --beta 3 \
    --learning_rate 2.5e-4 \
    --max_grad_norm 0.5 \
    --kl_regularization_method 'clip' \
    --clip_ratio 0.1 \
    --entropy_coef 0.01 \
    --vf_coef 0.5 \
    --joint_network 'True' \
    --seed 42 \
    --save_best_model 'True' \
    --min_save_interval_seconds 120 \
    --restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.
done


echo "Done with all Atari tests part 2!"