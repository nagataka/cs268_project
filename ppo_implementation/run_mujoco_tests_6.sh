#!/usr/bin/env bash

for run in {1..3}
do
    OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
    OPENAI_LOGDIR=tests/half_cheetah_TRPO_1_env_run_${run}_$(date '+%d-%m-%Y_%H:%M:%S') \
    python -m baselines.run --alg=trpo_mpi --env='HalfCheetah-v2' --num_env 1 --num_timesteps 1e6 --seed $run
done