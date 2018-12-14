#!/usr/bin/env bash

for run in {1..3}
do
    OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
    OPENAI_LOGDIR=tests/half_cheetah_DDPG_1_env_run_${run}_$(date '+%d-%m-%Y_%H:%M:%S') \
    python -m baselines.ddpg.main --env-id 'HalfCheetah-v2' --num-timesteps 1e6 --evaluation --seed $run
done

echo "Done with Mujoco tests part 1!"


for run in {1..3}
do
    OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' \
    OPENAI_LOGDIR=tests/half_cheetah_A2C_1_env_run_${run}_$(date '+%d-%m-%Y_%H:%M:%S') \
    python -m baselines.run --alg=a2c --env='HalfCheetah-v2' --num_env 8 --num_timesteps 1e6 --seed $run
done
