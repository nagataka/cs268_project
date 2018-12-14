#!/usr/bin/env bash

python ppo.py \
--env_id 'BreakoutNoFrameskip-v4' \
--num_workers 8 \
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
--restore_from_checkpoint 'tests/breakout_ppo_default_params_12-12-2018_00:29:19/saved_model/model.ckpt' \
--do_demo 'True'