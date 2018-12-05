from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import tensorflow as tf
import numpy as np
import gym.spaces
from spinup.utils.logx import EpochLogger
from spinup.algos.ppo.core import count_vars
import time
import baselines.common.tf_util as U
from baselines.run import get_env_type
from baselines.common.cmd_util import make_atari_env, make_mujoco_env
from baselines.common.models import nature_cnn


def categorical_entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discounted_sum_with_dones(rewards, dones, gamma):
    # from baselines
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def run_ppo(envs, use_cnn=False, use_kl_penalty_instead_of_clip=False):

    if use_kl_penalty_instead_of_clip:
        print("kl penalty not implemented yet")
        exit(0)

    seed = 42
    tf.set_random_seed(seed)
    np.random.seed(seed)

    logger = EpochLogger()
    # Setup multiple envs in their own processes
    observations_are_continous = isinstance(envs.observation_space, gym.spaces.Box)
    actions_are_continous = isinstance(envs.action_space, gym.spaces.Box)
    logger.log("Observation space: {}".format(envs.observation_space))
    logger.log("Action space: {}".format(envs.action_space))

    # Build TensorFlow graph

    # Inputs to Actor Critic

    observations_tf_shape = (None, *envs.observation_space.shape) if observations_are_continous else (None, 1)
    observations_tf = tf.placeholder(dtype=tf.float32, shape=observations_tf_shape)

    actions_tf_shape = (None, *envs.action_space.shape) if actions_are_continous else (None,)
    actions_tf_type = tf.float32 if actions_are_continous else tf.int32
    actions_tf = tf.placeholder(dtype=actions_tf_type, shape=actions_tf_shape)

    if use_cnn:
        cnn_tf = nature_cnn(observations_tf)
    else:
        cnn_tf = None

    # Actor Network
    with tf.variable_scope("actor"):
        if use_cnn:
            actor_tf = cnn_tf
        else:
            actor_tf = observations_tf

        for hidden_size in [128, 128]:
            actor_tf = tf.layers.dense(actor_tf, units=hidden_size, activation=tf.nn.leaky_relu)

        actor_tf_output_layer_units = envs.action_space.shape[-1] if actions_are_continous else envs.action_space.n
        actor_tf = tf.layers.dense(actor_tf, actor_tf_output_layer_units, activation=None)

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

    with tf.variable_scope("critic"):
        # Value Network
        if use_cnn:
            critic_tf = cnn_tf
        else:
            critic_tf = observations_tf

        for hidden_size in [128, 128]:
            critic_tf = tf.layers.dense(critic_tf, hidden_size, activation=tf.nn.leaky_relu)
        value_tf = tf.squeeze(tf.layers.dense(critic_tf, 1, activation=None), axis=1)

    # Values Necessary for Loss Calculation
    advantages_tf = tf.placeholder(dtype=tf.float32)
    returns_tf = tf.placeholder(dtype=tf.float32)
    logp_old_tf = tf.placeholder(dtype=tf.float32, shape=(None,))
    assert logp_old_tf.shape.as_list() == logp_tf.shape.as_list()

    clip_ratio = 0.2
    entropy_coef = 0.01

    approx_kl = tf.reduce_mean(logp_old_tf - logp_tf)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp_tf)  # a sample estimate for entropy, also easy to compute

    policy_ratio = tf.exp(logp_tf - logp_old_tf)
    min_adv = tf.where(advantages_tf > 0, (1 + clip_ratio) * advantages_tf, (1 - clip_ratio) * advantages_tf)
    pi_loss = -tf.reduce_mean(tf.minimum(policy_ratio * advantages_tf, min_adv)) - entropy_coef * approx_ent
    v_loss = tf.reduce_mean((returns_tf - value_tf) ** 2)

    # Info (useful to watch during learning)

    clipped = tf.logical_or(policy_ratio > (1 + clip_ratio), policy_ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    actor_lr = 3e-4
    critic_lr = 1e-3

    train_pi = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(loss=pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(loss=v_loss)

    # TODO: remove two lines
    var_counts = tuple(count_vars(scope) for scope in ['actor', 'critic'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    with sess.as_default():

        steps_per_epoch = 128 * num_procs
        epochs = 10000
        gamma = 0.99
        lam = 0.97
        train_pi_iters = 80
        train_v_iters = 80
        target_kl = 0.01
        local_steps_per_epoch = int(steps_per_epoch / num_procs)

        curr_obs = envs.reset()

        curr_dones = np.full(num_procs, False)

        curr_episode_returns = np.zeros(num_procs)
        curr_episode_lengths = np.zeros(num_procs)

        start_time = time.time()

        observations_input_shape = (-1, *envs.observation_space.shape) if observations_are_continous else (-1, 1)

        for epoch in range(epochs):

            # any_episodes_were_finished = False

            batch_obs = []
            batch_rewards = []
            batch_actions = []
            batch_logps = []
            batch_values = []
            batch_dones = []

            for t in range(local_steps_per_epoch):
                curr_actions, curr_values, curr_logps = sess.run([pi_tf, value_tf, logp_pi_tf],
                                                  feed_dict={observations_tf: np.reshape(curr_obs, observations_input_shape)})

                #TODO REMOVE LINE!!!!!!!!!!!!!11
                # curr_actions = [1]

                for val in curr_values:
                    logger.store(VVals=val)
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
                        print("Episode finished {} reward {} steps".format(curr_episode_returns[i], curr_episode_lengths[i]))
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


            batch_size = num_procs*local_steps_per_epoch
            final_batch_obs_shape = (batch_size,) + (envs.observation_space.shape if observations_are_continous else (1,))
            batch_obs = batch_obs.reshape(final_batch_obs_shape)

            final_batch_actions_shape = (batch_size,) + envs.action_space.shape if actions_are_continous else (-1,)
            batch_actions = batch_actions.reshape(final_batch_actions_shape)

            batch_returns = batch_returns.reshape((batch_size,))
            batch_advantages = batch_advantages.reshape((batch_size,))
            # batch_values = batch_values.reshape((-1, 1))
            batch_logps = batch_logps.reshape((batch_size,))

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

            pi_l_old, v_l_old, entropy = sess.run([pi_loss, v_loss, approx_ent], feed_dict=train_inputs)

            for i in range(train_pi_iters):
                _, kl, new_logp, min_adv_calc, policy_ratio_calc = sess.run([train_pi, approx_kl, logp_tf, min_adv, policy_ratio], feed_dict=train_inputs)
                # logger.log("new logp: {}".format(new_logp))
                # logger.log("min adv: {}".format(min_adv_calc))
                # logger.log("policy ratio: {}".format(policy_ratio_calc))
                if kl > 1.5 * target_kl:
                    logger.log("early stopping at step {} due to reaching max kl".format(i))
                    break
            logger.store(StopIter=i)

            critic_vars_before_update = U.GetFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))()

            for _ in range(train_v_iters):
                sess.run(train_v, feed_dict=train_inputs)

            critic_vars_after_update = U.GetFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))()

            assert len(critic_vars_before_update) == len(critic_vars_after_update)
            assert len(critic_vars_before_update) > 100
            assert not np.allclose(critic_vars_before_update, critic_vars_after_update)

            pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=train_inputs)
            logger.store(LossPi=pi_l_old, LossV=v_l_old,
                         KL=kl, Entropy=entropy, ClipFrac=cf,
                         DeltaLossPi=(pi_l_new - pi_l_old), DeltaLossV=(v_l_new - v_l_old))

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            # if any_episodes_were_finished:
            #     logger.log_tabular('EpRet', with_min_and_max=True)
            #     logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            # exit()


if __name__ == '__main__':

    env_id = "PongNoFrameskip-v4"
    num_procs = 8
    seed = 42
    use_kl_penalty = False

    env_type = get_env_type(env_id)[0]

    if env_type == 'atari':
        print("Atari environment detected")
        envs = make_atari_env(env_id, num_procs, seed)
    elif env_type == 'mujoco':
        print("Mujoco environment detected")
        envs = SubprocVecEnv([lambda: make_mujoco_env(env_id, seed + i if seed is not None else None, 1) for i in range(num_procs)])
    else:
        print("No specific environment type detected")
        env_type = None
        envs = SubprocVecEnv([lambda: gym.make(env_id) for _ in range(num_procs)])

    run_ppo(envs=envs, use_cnn=True if env_type == 'atari' else False, use_kl_penalty_instead_of_clip=use_kl_penalty)
