# CS268 Fall 2018 Final Project
# J.B. Lanier and Takashi Nagata

import tensorflow as tf
import numpy as np
import gym.spaces
from gym.wrappers import Monitor
import time
import argparse
import os
import cv2
from collections import deque

from baselines import logger

# Gym Environment wrappers
from baselines.run import get_env_type
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.cmd_util import make_atari_env, make_mujoco_env

# Network Architectures for fair comparison with other algorithms
from baselines.common.models import nature_cnn, mlp

EPSILON = 1e-8

DEFAULT_PARAMS = {
    "env_id": "PongNoFrameskip-v4",
    "num_workers": 32,
    "total_timesteps": 2e7,
    "num_train_updates_per_epoch": 4,
    "num_minibatches_per_epoch_train_update": 4,
    "time_horizon": 128,
    "gamma": 0.99,
    "lam": 0.95,
    "target_kl": 0.01,
    "beta": 3,
    "learning_rate": 2.5e-4,
    "max_grad_norm": 0.5,
    "kl_regularization_method": "clip",  # options are 'clip', 'penalty', or 'adaptive-penalty'
    "clip_ratio": 0.1,
    "entropy_coef": 0.01,
    "vf_coef": 0.5,
    "joint_network": True,
    "seed": 42,
    "save_best_model": True,
    "min_save_interval_seconds": 120,
    "restore_from_checkpoint": 'none',
    "do_demo": False
}


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def get_convert_arg_to_type_fn(arg_type):

    if arg_type == bool:
        def fn(value):
            if value in ['None', 'none']:
                return None
            if value in ['True', 'true', 't', '1']:
                return True
            elif value in ['False', 'false', 'f', '0']:
                return False
            else:
                raise ValueError("Argument must either be the string, \'True\' or \'False\'")
        return fn

    elif arg_type == int:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return int(float(value))
        return fn
    elif arg_type == str:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return str(value)
        return fn
    else:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return arg_type(value)
        return fn


def gaussian_likelihood(x, mu, log_std):
    # from spinning up
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discounted_sum_with_dones(rewards, dones, gamma):
    # from baselines A2C
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def run_ppo(envs, use_cnn, num_workers, total_timesteps, num_train_updates_per_epoch,
            num_minibatches_per_epoch_train_update, time_horizon, gamma, lam, target_kl, beta,
            learning_rate, max_grad_norm, kl_regularization_method, clip_ratio,
            entropy_coef, vf_coef, joint_network, seed, save_best_model, min_save_interval_seconds, restore_from_checkpoint, do_demo, **kwargs):

    print("saved_args is", locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    observations_are_continous = isinstance(envs.observation_space, gym.spaces.Box)
    actions_are_continous = isinstance(envs.action_space, gym.spaces.Box)
    logger.info("Observation space: {}".format(envs.observation_space))
    logger.info("Action space: {}".format(envs.action_space))

    # Build TensorFlow graph

    observations_tf_shape = (None, *envs.observation_space.shape) if observations_are_continous else (None, 1)
    observations_tf = tf.placeholder(dtype=tf.float32, shape=observations_tf_shape)

    actions_tf_shape = (None, *envs.action_space.shape) if actions_are_continous else (None,)
    actions_tf_type = tf.float32 if actions_are_continous else tf.int32
    actions_tf = tf.placeholder(dtype=actions_tf_type, shape=actions_tf_shape)

    if kl_regularization_method == 'adaptive-penalty':
        beta_tf = tf.Variable(initial_value=beta, trainable=False, dtype=tf.float32) # to compute KL penalty
    elif kl_regularization_method == 'penalty':
        beta_tf = tf.constant(beta, dtype=tf.float32)
    elif kl_regularization_method == 'clip':
        beta_tf = tf.no_op()

    with tf.variable_scope("actor_critic"):
        if use_cnn:
            if joint_network:
                actor_tf = value_tf = nature_cnn(observations_tf)
            else:
                with tf.variable_scope("actor"):
                    actor_tf = nature_cnn(observations_tf)
                with tf.variable_scope("critic"):
                    value_tf = nature_cnn(observations_tf)
        else:
            if joint_network:
                ac_tf, _ = mlp()(observations_tf)
                actor_tf = value_tf = ac_tf
            else:
                with tf.variable_scope("actor"):
                    actor_tf, _ = mlp()(observations_tf)
                with tf.variable_scope("critic"):
                    value_tf, _ = mlp()(observations_tf)

        actor_tf_output_layer_units = envs.action_space.shape[-1] if actions_are_continous else envs.action_space.n
        actor_tf = tf.layers.dense(actor_tf, actor_tf_output_layer_units, activation=None)

        value_tf = tf.squeeze(tf.layers.dense(value_tf, 1, activation=None), axis=1)

        if actions_are_continous:
            log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(envs.action_space.shape[-1], dtype=np.float32))
            std = tf.exp(log_std)
            pi_tf = actor_tf + tf.random_normal(tf.shape(actor_tf)) * std
            logp_tf = gaussian_likelihood(actions_tf, actor_tf, log_std)
            logp_pi_tf = gaussian_likelihood(pi_tf, actor_tf, log_std)
        else:
            logp_all = tf.nn.log_softmax(actor_tf)
            pi_tf = tf.squeeze(tf.multinomial(actor_tf, 1), axis=1)
            logp_tf = tf.reduce_sum(tf.one_hot(actions_tf, depth=envs.action_space.n) * logp_all, axis=1)
            logp_pi_tf = tf.reduce_sum(tf.one_hot(pi_tf, depth=envs.action_space.n) * logp_all, axis=1)

        assert logp_pi_tf.shape.as_list() == logp_tf.shape.as_list()

    advantages_tf = tf.placeholder(dtype=tf.float32)
    returns_tf = tf.placeholder(dtype=tf.float32)
    logp_old_tf = tf.placeholder(dtype=tf.float32, shape=(None,))
    value_old_tf = tf.placeholder(dtype=tf.float32, shape=(None,))
    clip_ratio_tf = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate_tf = tf.placeholder(dtype=tf.float32, shape=())
    assert logp_old_tf.shape.as_list() == logp_tf.shape.as_list()

    approx_kl_tf = tf.reduce_mean(logp_old_tf - logp_tf)  # a sample estimate for KL-divergence, easy to compute
    approx_ent_tf = tf.reduce_mean(-logp_tf)  # a sample estimate for entropy, also easy to compute

    policy_ratio_tf = tf.exp(logp_tf - logp_old_tf)

    if kl_regularization_method == 'clip':
        clipped_pi_loss_tf = advantages_tf * tf.clip_by_value(policy_ratio_tf, 1.0 - clip_ratio_tf, 1.0 + clip_ratio_tf)
        pi_loss_tf = -tf.reduce_mean(tf.minimum(policy_ratio_tf * advantages_tf, clipped_pi_loss_tf))
    elif 'penalty' in kl_regularization_method:
        pi_loss_tf = -tf.reduce_mean(policy_ratio_tf * advantages_tf - beta_tf * approx_kl_tf)
    v_loss_tf = 0.5 * tf.reduce_mean((returns_tf - value_tf) ** 2)
    loss_tf = pi_loss_tf + vf_coef * v_loss_tf - entropy_coef * approx_ent_tf

    if kl_regularization_method == 'adaptive-penalty':
        new_beta_tf = tf.case(
            {tf.less(approx_kl_tf, target_kl / 1.5): lambda: (beta_tf / 2) + EPSILON,
             tf.greater(approx_kl_tf, target_kl * 1.5): lambda: beta_tf * 2},
            default=lambda: beta_tf,
            exclusive=True
        )
        update_beta_if_applicable_op = tf.assign(beta_tf, new_beta_tf)
    else:
        update_beta_if_applicable_op = tf.no_op()

    clipped_tf = tf.logical_or(policy_ratio_tf > (1 + clip_ratio_tf), policy_ratio_tf < (1 - clip_ratio_tf))
    clipfrac_tf = tf.reduce_mean(tf.cast(clipped_tf, tf.float32))

    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf, epsilon=1e-5)
    params = tf.trainable_variables('actor_critic')
    grads_and_var = trainer.compute_gradients(loss_tf, params)
    grads, var = zip(*grads_and_var)

    if max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads_and_var = list(zip(grads, var))

    train_op = trainer.apply_gradients(grads_and_var)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if restore_from_checkpoint is not None:
        logger.info("Restoring model from {}".format(restore_from_checkpoint))
        saver.restore(sess, restore_from_checkpoint)

    sess.graph.finalize()
    logger.log_graph_to_tensorboard(sess.graph)

    observations_input_shape = (-1, *envs.observation_space.shape) if observations_are_continous else (-1, 1)

    with sess.as_default():

        if do_demo:

            curr_obs = envs.reset()
            # frame = envs.unwrapped.render(mode='rgb_array')
            # print("frame shape: {}".format(frame.shape))
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # out = cv2.VideoWriter('output.avi', fourcc, 60, (frame.shape[1], frame.shape[0]), True)

            try:
                while True:
                    curr_actions, _, _ = sess.run([pi_tf, value_tf, logp_pi_tf],
                                            feed_dict={observations_tf: np.reshape(curr_obs, observations_input_shape)})
                    curr_obs, _, dones, infos = envs.step(curr_actions)
                    envs.render()
                    # frame = envs.unwrapped.unwrapped.render(mode='rgb_array').copy()
                    # cv2.waitKey(1)
                    #
                    # cv2.imshow("frame", frame)
                    # out.write(frame)

                    for info in infos:
                        maybeepinfo = info.get('episode')
                        if maybeepinfo:
                            print("episode length: {} reward: {}".format(maybeepinfo['l'],maybeepinfo['r']))
            except KeyboardInterrupt:
                # out.release()
                # cv2.destroyAllWindows()
                # print('video saved')
                exit()

        total_timesteps = int(total_timesteps)
        total_epoch_batch_size = time_horizon * num_workers
        num_epochs = total_timesteps // total_epoch_batch_size
        train_minibatch_size = total_epoch_batch_size // num_minibatches_per_epoch_train_update

        curr_obs = envs.reset()
        curr_dones = np.full(num_workers, False)

        start_time = time.time()

        episode_lengths = deque(maxlen=100)
        episode_returns = deque(maxlen=100)

        best_mean_episode_return_so_far = -np.inf
        last_save_time = None

        for epoch in range(1, num_epochs+1):

            fraction_of_epochs_left = 1.0 - (epoch - 1.0) / num_epochs

            # any_episodes_were_finished = False

            batch_obs = []
            batch_rewards = []
            batch_actions = []
            batch_logps = []
            batch_values = []
            batch_dones = []

            rollout_values = []

            for t in range(time_horizon):
                curr_actions, curr_values, curr_logps = sess.run([pi_tf, value_tf, logp_pi_tf],
                                                  feed_dict={observations_tf: np.reshape(curr_obs, observations_input_shape)})

                for val in curr_values:
                    rollout_values.append(val)

                batch_obs.append(curr_obs.copy())
                batch_dones.append(curr_dones)

                batch_actions.append(curr_actions)
                batch_values.append(curr_values)
                batch_logps.append(curr_logps)

                curr_obs, curr_rewards, curr_dones, infos = envs.step(curr_actions)
                batch_rewards.append(curr_rewards)

                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        episode_lengths.append(maybeepinfo['l'])
                        episode_returns.append(maybeepinfo['r'])

            batch_dones.append(curr_dones)

            # Make first axis episode and second axis step

            batch_obs = np.asarray(batch_obs, dtype=np.float32).swapaxes(1, 0)
            batch_rewards = np.asarray(batch_rewards, dtype=np.float32).swapaxes(1, 0)
            batch_actions = np.asarray(batch_actions).swapaxes(1, 0)
            batch_logps = np.asarray(batch_logps, dtype=np.float32).swapaxes(1, 0)
            batch_values = np.asarray(batch_values, dtype=np.float32).swapaxes(1, 0)
            batch_dones = np.asarray(batch_dones, dtype=np.bool).swapaxes(1, 0)
            batch_dones = batch_dones[:, 1:]

            batch_advantages = np.empty_like(batch_rewards)

            # discount/bootstrap off value fn
            last_values = sess.run(value_tf, feed_dict={observations_tf: np.reshape(curr_obs, observations_input_shape)})
            for n, (rewards, dones, values, last_value) in enumerate(zip(batch_rewards, batch_dones, batch_values, last_values)):

                rewards = rewards.tolist()
                if dones[-1] == 0:
                    values = np.append(values, last_value)
                    rewards = np.append(rewards, last_value)
                else:
                    values = np.append(values, 0)
                    rewards = np.append(rewards, 0)
                deltas = (rewards[:-1] + gamma * values[1:] * (1. - dones) - values[:-1])
                dones = dones.tolist()

                # assert np.array_equal(np.shape(deltas), np.shape(dones))
                advantages = discounted_sum_with_dones(deltas, dones, (gamma * lam))

                # returns = advantages + values[:-1]

                # assert np.array_equal(returns, advantages + values[:-1])

                batch_advantages[n] = advantages

            batch_returns = batch_values + batch_advantages

            final_batch_obs_shape = (total_epoch_batch_size,) + (envs.observation_space.shape if observations_are_continous else (1,))
            batch_obs = batch_obs.reshape(final_batch_obs_shape)

            final_batch_actions_shape = (total_epoch_batch_size,) + envs.action_space.shape if actions_are_continous else (total_epoch_batch_size,)
            batch_actions = batch_actions.reshape(final_batch_actions_shape)

            batch_returns = batch_returns.reshape((total_epoch_batch_size,))
            batch_advantages = batch_advantages.reshape((total_epoch_batch_size,))
            batch_values = batch_values.reshape((total_epoch_batch_size,))
            batch_logps = batch_logps.reshape((total_epoch_batch_size,))

            # Train actor critic

            train_inputs = {
                observations_tf: batch_obs,
                actions_tf: batch_actions,
                advantages_tf: batch_advantages,
                returns_tf: batch_returns,
                logp_old_tf: batch_logps,
                value_old_tf: batch_values,
            }

            sample_indexes = np.arange(total_epoch_batch_size)

            actor_losses = []
            critic_losses = []
            kls = []
            entropies = []
            clipfractions = []
            actor_loss_deltas = []
            critic_loss_deltas = []
            beta_vals = []

            for update_num in range(1, num_train_updates_per_epoch+1):
                np.random.shuffle(sample_indexes)

                kls_over_update = []

                debug_indexes_trained_on = []

                for i in range(num_minibatches_per_epoch_train_update):
                    start_index = i * train_minibatch_size
                    end_index = (i+1) * train_minibatch_size
                    samples_indexes_to_train_on = sample_indexes[start_index:end_index]

                    debug_indexes_trained_on.extend(samples_indexes_to_train_on)

                    minibatch_feed_dict = {k: v[samples_indexes_to_train_on] for k, v in train_inputs.items()}

                    batch_advantages = minibatch_feed_dict[advantages_tf].copy()
                    batch_advantages = (batch_advantages - np.mean(batch_advantages)) / (np.std(batch_advantages) + EPSILON)
                    minibatch_feed_dict[advantages_tf] = batch_advantages

                    minibatch_feed_dict[learning_rate_tf] = learning_rate * fraction_of_epochs_left
                    minibatch_feed_dict[clip_ratio_tf] = clip_ratio * fraction_of_epochs_left

                    pi_l_old, v_l_old, entropy = sess.run([pi_loss_tf, v_loss_tf, approx_ent_tf], feed_dict=minibatch_feed_dict)
                    _, _, kl, new_logp, policy_ratio_calc = sess.run([train_op, update_beta_if_applicable_op, approx_kl_tf, logp_tf, policy_ratio_tf], feed_dict=minibatch_feed_dict)
                    pi_l_new, v_l_new, kl, cf, beta_calc = sess.run([pi_loss_tf, v_loss_tf, approx_kl_tf, clipfrac_tf, beta_tf], feed_dict=minibatch_feed_dict)

                    beta_vals.append(beta_calc)
                    actor_losses.append(pi_l_old)
                    critic_losses.append(v_l_old)
                    kls_over_update.append(kl)
                    entropies.append(entropy)
                    clipfractions.append(cf)
                    actor_loss_deltas.append(pi_l_new - pi_l_old)
                    critic_loss_deltas.append(v_l_new - v_l_old)

                assert np.array_equal(np.sort(debug_indexes_trained_on), np.arange(0, total_epoch_batch_size))

                mean_kl_over_update = np.mean(kls_over_update)
                kls.append(mean_kl_over_update)

            mean_episode_return = np.nan if len(episode_returns) == 0 else np.mean(episode_returns)
            mean_epsiode_length = np.nan if len(episode_lengths) == 0 else np.mean(episode_lengths)

            logger.record_tabular("epoch", epoch)
            logger.record_tabular("stop_update_iter", update_num)

            if 'penalty' in kl_regularization_method:
                logger.record_tabular("beta", np.mean(beta_vals))
            if kl_regularization_method == 'adaptive-penalty':
                logger.record_tabular("kl_target", target_kl)
            logger.record_tabular("actor_loss", np.mean(actor_losses))
            logger.record_tabular("critic_loss", np.mean(critic_losses))
            logger.record_tabular("kl_div", np.mean(kls))
            logger.record_tabular("policy_entropy", np.mean(entropies))
            logger.record_tabular("clip _fracs", np.mean(clipfractions))
            logger.record_tabular("actor_loss_deltas", np.mean(actor_loss_deltas))
            logger.record_tabular("critic_loss_deltas", np.mean(critic_loss_deltas))
            logger.record_tabular("avg_rollout_value", np.mean(rollout_values))
            logger.record_tabular("avg_ep_return", mean_episode_return)
            logger.record_tabular("avg_ep_length", mean_epsiode_length)
            logger.record_tabular("time_steps", epoch * total_epoch_batch_size)
            logger.record_tabular("seconds_elapsed", time.time() - start_time)
            logger.dump_tabular()

            # Saving the graph if it's the best so far
            if mean_episode_return is not np.nan and mean_episode_return >= best_mean_episode_return_so_far and save_best_model:
                best_mean_episode_return_so_far = mean_episode_return
                save_path = os.path.join(logger.get_dir(), 'saved_model', 'model.ckpt')
                logger.info("New best mean episode return of {}".format(mean_episode_return))

                curr_time = time.time()
                if (last_save_time is None) or (curr_time - last_save_time >= min_save_interval_seconds):
                    last_save_time = curr_time
                    logger.info("Saving model to {}".format(save_path))
                    saver.save(sess, save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key
        parser.add_argument(key, type=get_convert_arg_to_type_fn(type(value)), default=value)

    args = parser.parse_args()
    dict_args = vars(args)

    logger.configure()

    log_params(dict_args)

    env_id = dict_args['env_id']
    num_workers = dict_args['num_workers']
    seed = dict_args['seed']

    env_type = get_env_type(env_id)[0]

    if env_type == 'atari':
        print("Atari environment detected")
        frame_stack_size = 4
        envs = VecFrameStack(make_atari_env(env_id, num_workers, seed), frame_stack_size)
    elif env_type == 'mujoco':
        print("Mujoco environment detected")
        if dict_args['do_demo']:
            dict_args['num_workers'] = 1
            envs = VecNormalize(DummyVecEnv([lambda: make_mujoco_env(env_id, seed)]))
        else:
            envs = VecNormalize(SubprocVecEnv([lambda: make_mujoco_env(env_id, seed + i if seed is not None else None, 1) for i in range(num_workers)]))
    else:
        print("No specific environment type detected")
        env_type = None
        envs = SubprocVecEnv([lambda: Monitor(gym.make(env_id), logger.get_dir() and os.path.join(logger.get_dir(), str(0) + '.' + str(rank))) for rank in range(num_workers)])

    assert dict_args['kl_regularization_method'] in ['clip', 'penalty', 'adaptive-penalty']

    run_ppo(envs=envs, use_cnn=True if env_type == 'atari' else False, **dict_args)
