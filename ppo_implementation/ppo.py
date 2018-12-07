from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import tensorflow as tf
import numpy as np
import gym.spaces
import time
import argparse
import os
from collections import deque
from baselines import logger
from baselines.run import get_env_type
from baselines.common.cmd_util import make_atari_env, make_mujoco_env
from baselines.common.models import nature_cnn

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
    "learning_rate": 2.5e-4,
    "max_grad_norm": 0.5,
    "kl_regularization_method": "clip",  # options are 'clip', 'penalty', or 'adaptive-penalty'
    "clip_ratio": 0.1,
    "entropy_coef": 0.01,
    "vf_coef": 0.5,
    "seed": 42,
    "save_best_model": True,
    "min_save_interval_seconds": 120,
    "restore_from_checkpoint": 'none'
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
            num_minibatches_per_epoch_train_update, time_horizon, gamma, lam, target_kl,
            learning_rate, max_grad_norm, kl_regularization_method, clip_ratio,
            entropy_coef, vf_coef, seed, save_best_model, min_save_interval_seconds, restore_from_checkpoint, **kwargs):

    if kl_regularization_method != 'clip':
        print("kl penalty method not implemented yet")
        exit(1)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Setup multiple envs in their own processes
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

    with tf.variable_scope("actor_critic"):
        if use_cnn:
            ac_tf = nature_cnn(observations_tf)
        else:
            ac_tf = observations_tf
            for hidden_size in [64, 64]:
                ac_tf = tf.layers.dense(ac_tf, units=hidden_size, activation=tf.nn.leaky_relu)

        actor_tf_output_layer_units = envs.action_space.shape[-1] if actions_are_continous else envs.action_space.n
        actor_tf = tf.layers.dense(ac_tf, actor_tf_output_layer_units, activation=None)

        value_tf = tf.squeeze(tf.layers.dense(ac_tf, 1, activation=None), axis=1)

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

    # Values Necessary for Loss Calculation
    advantages_tf = tf.placeholder(dtype=tf.float32)
    returns_tf = tf.placeholder(dtype=tf.float32)
    logp_old_tf = tf.placeholder(dtype=tf.float32, shape=(None,))
    assert logp_old_tf.shape.as_list() == logp_tf.shape.as_list()

    approx_kl = tf.reduce_mean(logp_old_tf - logp_tf)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp_tf)  # a sample estimate for entropy, also easy to compute

    policy_ratio = tf.exp(logp_tf - logp_old_tf)
    min_adv = tf.where(advantages_tf > 0, (1 + clip_ratio) * advantages_tf, (1 - clip_ratio) * advantages_tf)
    pi_loss = -tf.reduce_mean(tf.minimum(policy_ratio * advantages_tf, min_adv))
    v_loss = tf.reduce_mean((returns_tf - value_tf) ** 2)

    loss = pi_loss + vf_coef * v_loss - entropy_coef * approx_ent

    clipped = tf.logical_or(policy_ratio > (1 + clip_ratio), policy_ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    params = tf.trainable_variables('actor_critic')
    grads_and_var = trainer.compute_gradients(loss, params)
    grads, var = zip(*grads_and_var)

    if max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads_and_var = list(zip(grads, var))

    train = trainer.apply_gradients(grads_and_var)

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

    with sess.as_default():

        total_timesteps = int(total_timesteps)
        total_epoch_batch_size = time_horizon * num_workers
        num_epochs = total_timesteps // total_epoch_batch_size
        train_minibatch_size = total_epoch_batch_size // num_minibatches_per_epoch_train_update

        curr_obs = envs.reset()

        curr_dones = np.full(num_workers, False)

        curr_episode_returns = np.zeros(num_workers)
        curr_episode_lengths = np.zeros(num_workers)

        start_time = time.time()

        observations_input_shape = (-1, *envs.observation_space.shape) if observations_are_continous else (-1, 1)

        episode_lengths = deque(maxlen=100)
        episode_returns = deque(maxlen=100)

        best_mean_episode_return_so_far = -np.inf
        last_save_time = None

        for epoch in range(1, num_epochs+1):

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

                #TODO REMOVE LINE!!!!!!!!!!!!!11
                # curr_actions = [1]

                for val in curr_values:
                    rollout_values.append(val)
                batch_obs.append(curr_obs)
                batch_actions.append(curr_actions)
                batch_logps.append(curr_logps)
                batch_values.append(curr_values)
                batch_dones.append(curr_dones)

                curr_obs, curr_rewards, curr_dones, _ = envs.step(curr_actions)
                batch_rewards.append(curr_rewards)
                # logger.log(curr_rewards)

                curr_episode_returns += curr_rewards
                curr_episode_lengths += 1
                for i, done in enumerate(curr_dones):
                    if done:
                        episode_lengths.append(curr_episode_lengths[i])
                        episode_returns.append(curr_episode_returns[i])
                        # print("Episode finished {} reward {} steps".format(curr_episode_returns[i], curr_episode_lengths[i]))
                        # any_episodes_were_finished = True
                        # logger.store(EpRet=curr_episode_returns[i], EpLen=curr_episode_lengths[i])
                        curr_episode_returns[i] = 0
                        curr_episode_lengths[i] = 0

            batch_dones.append(curr_dones)

            # Make first axis episode and second axis step

            batch_obs = np.asarray(batch_obs, dtype=np.float32).swapaxes(1, 0)
            batch_rewards = np.asarray(batch_rewards, dtype=np.float32).swapaxes(1, 0)
            batch_actions = np.asarray(batch_actions).swapaxes(1, 0)
            batch_logps = np.asarray(batch_logps, dtype=np.float32).swapaxes(1, 0)
            batch_values = np.asarray(batch_values, dtype=np.float32).swapaxes(1, 0)
            batch_dones = np.asarray(batch_dones, dtype=np.bool).swapaxes(1, 0)
            batch_dones = batch_dones[:, 1:]

            batch_returns = np.empty_like(batch_rewards)
            batch_advantages = np.empty_like(batch_rewards)

            # discount/bootstrap off value fn
            last_values = sess.run(value_tf, feed_dict={observations_tf: np.reshape(curr_obs, observations_input_shape)})
            for n, (rewards, dones, values, last_value) in enumerate(zip(batch_rewards, batch_dones, batch_values, last_values)):

                #TODO REMOVE THESE TWO LINES!!!!!
                # values = [0.01081378, -0.04990983, -0.11157902]
                # last_value = -0.17283736

                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    returns = discounted_sum_with_dones(rewards + [last_value], dones + [0], gamma)[:-1]
                    values = np.append(values, last_value)
                    rewards = np.append(rewards, last_value)
                else:
                    returns = discounted_sum_with_dones(rewards, dones, gamma)
                    values = np.append(values, 0)
                    rewards = np.append(rewards, 0)

                deltas = (rewards[:-1] + gamma * values[1:] - values[:-1])

                # print("deltas: {}".format(deltas))

                advantages = discounted_sum_with_dones(deltas, dones + [0], (gamma * lam))
                advantages = (advantages - np.mean(advantages)) / np.std(advantages)

                batch_returns[n] = returns
                batch_advantages[n] = advantages

            final_batch_obs_shape = (total_epoch_batch_size,) + (envs.observation_space.shape if observations_are_continous else (1,))
            batch_obs = batch_obs.reshape(final_batch_obs_shape)

            final_batch_actions_shape = (total_epoch_batch_size,) + envs.action_space.shape if actions_are_continous else (-1,)
            batch_actions = batch_actions.reshape(final_batch_actions_shape)

            batch_returns = batch_returns.reshape((total_epoch_batch_size,))
            batch_advantages = batch_advantages.reshape((total_epoch_batch_size,))
            # batch_values = batch_values.reshape((-1, 1))
            batch_logps = batch_logps.reshape((total_epoch_batch_size,))

            # Train actor critic

            train_inputs = {
                observations_tf: batch_obs,
                actions_tf: batch_actions,
                advantages_tf: batch_advantages,
                returns_tf: batch_returns,
                logp_old_tf: batch_logps
            }

            # debug_dict = {
            #     "observations": batch_obs,
            #     "actions": batch_actions,
            #     "rewards": batch_rewards,
            #     "dones": batch_dones,
            #     "values": batch_values,
            #     "advantages": batch_advantages,
            #     "returns": batch_returns,
            #     "logp_old": batch_logps
            # }
            # logger.log(debug_dict)

            sample_indexes = np.arange(total_epoch_batch_size)

            actor_losses = []
            critic_losses = []
            kls = []
            entropies = []
            clipfractions = []
            actor_loss_deltas = []
            critic_loss_deltas = []

            for update_num in range(1, num_train_updates_per_epoch+1):
                np.random.shuffle(sample_indexes)

                kls_over_update = []

                for i in range(num_minibatches_per_epoch_train_update):
                    start_index = i * train_minibatch_size
                    end_index = (i+1) * train_minibatch_size
                    samples_indexes_to_train_on = sample_indexes[start_index:end_index]
                    minibatch_feed_dict = {k: v[samples_indexes_to_train_on] for k, v in train_inputs.items()}

                    pi_l_old, v_l_old, entropy = sess.run([pi_loss, v_loss, approx_ent], feed_dict=minibatch_feed_dict)

                    _, kl, new_logp, min_adv_calc, policy_ratio_calc = sess.run([train, approx_kl, logp_tf, min_adv, policy_ratio], feed_dict=minibatch_feed_dict)

                    pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=minibatch_feed_dict)

                    actor_losses.append(pi_l_old)
                    critic_losses.append(v_l_old)
                    kls_over_update.append(kl)
                    entropies.append(entropy)
                    clipfractions.append(cf)
                    actor_loss_deltas.append(pi_l_new - pi_l_old)
                    critic_loss_deltas.append(v_l_new - v_l_old)

                mean_kl_over_update = np.mean(kls_over_update)
                kls.append(mean_kl_over_update)

                if np.mean(mean_kl_over_update) > 1.5 * target_kl:
                    logger.info("early stopping at step {} due to reaching max kl".format(i))
                    break

            mean_episode_return = np.nan if len(episode_returns) == 0 else np.mean(episode_returns)
            mean_epsiode_length = np.nan if len(episode_lengths) == 0 else np.mean(episode_lengths)

            logger.record_tabular("epoch", epoch)
            logger.record_tabular("stop_update_iter", update_num)
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

            if mean_episode_return is not np.nan and mean_episode_return >= best_mean_episode_return_so_far and save_best_model:
                best_mean_episode_return_so_far = mean_episode_return
                save_path = os.path.join(logger.get_dir(), 'saved_model', 'model.ckpt')
                logger.info("New best mean episode return of {}".format(mean_episode_return))

                curr_time = time.time()
                if (last_save_time is None) or (curr_time - last_save_time >= min_save_interval_seconds):
                    last_save_time = curr_time
                    logger.info("Saving model to {}".format(save_path))
                    saver.save(sess, save_path)

        # for i in range(train_pi_iters):
            #     # logger.log("new logp: {}".format(new_logp))
            #     # logger.log("min adv: {}".format(min_adv_calc))
            #     # logger.log("policy ratio: {}".format(policy_ratio_calc))


            # critic_vars_before_update = U.GetFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))()

            # critic_vars_after_update = U.GetFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))()
            #
            # assert len(critic_vars_before_update) == len(critic_vars_after_update)
            # assert len(critic_vars_before_update) > 100
            # assert not np.allclose(critic_vars_before_update, critic_vars_after_update)


            #
            # # # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # # # if any_episodes_were_finished:
            # # #     logger.log_tabular('EpRet', with_min_and_max=True)
            # # #     logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            # logger.log_tabular('TotalEnvInteracts', epoch * total_epoch_batch_size)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('DeltaLossPi', average_only=True)
            # logger.log_tabular('DeltaLossV', average_only=True)
            # logger.log_tabular('Entropy', average_only=True)
            # logger.log_tabular('KL', average_only=True)
            # logger.log_tabular('ClipFrac', average_only=True)
            # # logger.log_tabular('StopIter', average_only=True)
            # logger.log_tabular('Time', time.time() - start_time)
            # logger.dump_tabular()

            # exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key
        # key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=get_convert_arg_to_type_fn(type(value)), default=value)

    args = parser.parse_args()
    dict_args = vars(args)

    logger.configure()

    # dict_args = prepare_params(dict_args)
    log_params(dict_args)

    # env_id = "CartPole-v1"
    # env_id = "HalfCheetah-v2"

    env_id = dict_args['env_id']
    num_workers = dict_args['num_workers']
    seed = dict_args['seed']

    env_type = get_env_type(env_id)[0]

    if env_type == 'atari':
        print("Atari environment detected")
        envs = make_atari_env(env_id, num_workers, seed)
    elif env_type == 'mujoco':
        print("Mujoco environment detected")
        envs = SubprocVecEnv([lambda: make_mujoco_env(env_id, seed + i if seed is not None else None, 1) for i in range(num_workers)])
    else:
        print("No specific environment type detected")
        env_type = None
        envs = SubprocVecEnv([lambda: gym.make(env_id) for _ in range(num_workers)])

    run_ppo(envs=envs, use_cnn=True if env_type == 'atari' else False, **dict_args)
