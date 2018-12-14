#!/usr/bin/env bash


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_TRPO_8_env_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=trpo_mpi --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7 --seed 432


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_A2C_8_envs_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=a2c --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7 --seed 432


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_TRPO_8_env_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=trpo_mpi --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7 --seed 4142


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_A2C_8_envs_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=a2c --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7 --seed 4142


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_TRPO_8_env_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=trpo_mpi --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7 --seed 41422


OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
OPENAI_LOGDIR=tests/breakout_A2C_8_envs_$(date '+%d-%m-%Y_%H:%M:%S') \
python -m baselines.run --alg=a2c --env='BreakoutNoFrameskip-v4' --num_env 8 --num_timesteps 1e7 --seed 3425