#!/usr/bin/env bash

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_ppo_default_params_$(date '+%d-%m-%Y_%H:%M:%S') \
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
--seed 4432 \
--save_best_model 'True' \
--min_save_interval_seconds 120 \
--restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.



OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_ppo_no_entropy_term_$(date '+%d-%m-%Y_%H:%M:%S') \
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
--entropy_coef 0.0 \
--vf_coef 0.5 \
--joint_network 'True' \
--seed 42432 \
--save_best_model 'True' \
--min_save_interval_seconds 120 \
--restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.




OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_ppo_seperate_networks_$(date '+%d-%m-%Y_%H:%M:%S') \
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
--joint_network 'False' \
--seed 44322 \
--save_best_model 'True' \
--min_save_interval_seconds 120 \
--restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.






OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_ppo_default_params_$(date '+%d-%m-%Y_%H:%M:%S') \
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
--seed 8732 \
--save_best_model 'True' \
--min_save_interval_seconds 120 \
--restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.



OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_ppo_no_entropy_term_$(date '+%d-%m-%Y_%H:%M:%S') \
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
--entropy_coef 0.0 \
--vf_coef 0.5 \
--joint_network 'True' \
--seed 428732 \
--save_best_model 'True' \
--min_save_interval_seconds 120 \
--restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.




OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_ppo_seperate_networks_$(date '+%d-%m-%Y_%H:%M:%S') \
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
--joint_network 'False' \
--seed 44562 \
--save_best_model 'True' \
--min_save_interval_seconds 120 \
--restore_from_checkpoint 'none' # During training, when the model is saved, the checkpoint location will be listed. Paste that here.
