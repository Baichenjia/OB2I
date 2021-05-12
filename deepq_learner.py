import tensorflow as tf
import numpy as np
from models import QNetwork, BootQNetwork, BootQNetworkWithPrior
from baselines.common.running_mean_std import RunningMeanStd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


class DEEPQ(tf.keras.Model):
    def __init__(self, observation_shape, num_actions, optimizer, num_ensemble, normrew, normnxq, normrew_ebu, normnxq_ebu,
                 prior, prior_scale, ebu, beta, batch_size, grad_norm_clipping=None, gamma=1.0, gradient_norm=True,
                 double_q=False, param_noise=False, param_noise_filter_func=None):
        super(DEEPQ, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.double_q = double_q
        self.param_noise = param_noise
        self.param_noise_filter_func = param_noise_filter_func
        self.grad_norm_clipping = grad_norm_clipping
        self.optimizer = optimizer
        self.num_ensemble = num_ensemble
        self.gradient_norm = gradient_norm
        self.batch_size = batch_size
        self.randomized_prior = prior
        self.prior_scale = prior_scale
        self.normrew = normrew                    # if norm the bonus of immediate rewards
        self.normnxq = normnxq                    # if norm the bonus of next-Q values
        self.normrew_ebu = normrew_ebu,
        self.normnxq_ebu = normnxq_ebu,
        self.ebu = ebu
        self.beta = beta
        self.batch_size = batch_size

        if self.ebu:                              # store episodic values in EBU-update
            self.batchnum = 0
            self.epi_len = 0
            self.batch_count = 0
            self.epi_state = None
            self.epi_actions = None
            self.epi_rewards = None
            self.epi_terminals = None
            self.Q_tilde = None
            self.y_ = None
            if self.normrew_ebu:                      # if norm the reward
                self.rff_rms = RunningMeanStd()       # norm the bonus of immediate reward
            if self.normnxq_ebu:
                self.rff_rms_q = RunningMeanStd()     # norm the bonus of target-Q
        else:
            if self.normrew:                          
                self.rff_rms = RunningMeanStd()       # norm the bonus of immediate reward
            if self.normnxq:
                self.rff_rms_q = RunningMeanStd()     # norm the bonus of target-Q

        # Randomized prior function (osband. 2018)
        if self.randomized_prior:
            with tf.name_scope('prior'):              # observation_shape = (None,84,84,4)
                self.prior_network = BootQNetwork(num_actions=num_actions, num_ensemble=num_ensemble)
            with tf.name_scope('q_network'):          # observation_shape = (None,84,84,4)
                self.q_network_main = BootQNetwork(num_actions=num_actions, num_ensemble=num_ensemble)
                self.q_network = BootQNetworkWithPrior(
                    prior=self.prior_network, main=self.q_network_main, prior_scale=self.prior_scale)
            with tf.name_scope('target_q_network'):
                self.target_q_network_main = BootQNetwork(num_actions=num_actions, num_ensemble=num_ensemble)
                self.target_q_network = BootQNetworkWithPrior(
                    prior=self.prior_network, main=self.target_q_network_main, prior_scale=self.prior_scale)
        else:
            with tf.name_scope('q_network'):          # observation_shape = (None,84,84,4)
                self.q_network = BootQNetwork(num_actions=num_actions, num_ensemble=num_ensemble)
            with tf.name_scope('target_q_network'):
                self.target_q_network = BootQNetwork(num_actions=num_actions, num_ensemble=num_ensemble)
        self.eps = tf.Variable(0., name="eps")

    def step(self, obs, active_head=None, stochastic=True, update_eps=-1.0, action_selection='sample',
             action_ucb_ratio=0.1, action_mcb_ratio=0.2, action_ids_ratio=0.1):
        # active_head indicate the index of Q-head in prediction
        assert action_selection in ['sample', 'ucb', 'mcb', 'vote', 'ids', 'mean']
        if action_selection == 'vote':
            action_values = self.q_network(obs, k=None)                    # (1, 10, n_actions)
            top_action_votings = tf.argmax(action_values, axis=-1)         # (1, 10)
            counts = np.bincount(top_action_votings[0].numpy())            # (num_actions,)
            top_action = np.argmax(counts)                                 # vote for decision.
            # convert the top action to a one hot vector
            q_values = tf.convert_to_tensor(np.eye(self.num_actions)[[top_action]], tf.float32)  # (1, num_actions)
        elif action_selection == 'ucb':
            action_values = self.q_network(obs, k=None)                    # (1, 10, n_actions)
            action_values_mean = tf.reduce_mean(action_values, axis=1)     # (1, n_actions)
            action_values_std = tf.math.reduce_std(action_values, axis=1)  # (1, n_actions)
            q_values = action_values_mean + action_ucb_ratio * action_values_std  # (1, n_actions)
        elif action_selection == 'mcb':
            action_values = self.q_network(obs, k=None)                                  # (1, 10, n_actions)
            action_values_mean = tf.reduce_mean(action_values, axis=1, keepdims=True)    # (1, 1,  n_actions)
            action_value_mcb = tf.maximum(action_values - action_values_mean, 0.)        # (1, 10, n_actions)
            action_value_mcb = tf.reduce_mean(action_value_mcb, axis=1)                  # (1, n_actions)
            q_values = tf.squeeze(action_values_mean, 1) + action_mcb_ratio * action_value_mcb  # (1, n_actions)
        elif action_selection == 'ids':
            action_values = self.q_network(obs, k=None)                # (1, 10, n_actions)
            mean = tf.reduce_mean(action_values, axis=1)               # mean (None, n_action)
            zero_mean = action_values - tf.expand_dims(mean, axis=-2)  # zero_mean (None, 10, n_action)
            var = tf.reduce_mean(tf.square(zero_mean), axis=1)         # var (None, n_action)
            std = tf.sqrt(var)                                         # std (None, n_action)
            regret = tf.reduce_max(mean + action_ids_ratio * std, axis=-1, keepdims=True)  #
            regret = regret - (mean - action_ids_ratio * std)          # regret (None, n_action)
            regret_sq = tf.square(regret)                              # regret_sq (None, n_action)
            info_gain = tf.log(1 + var / 1.0) + 1e-5                   # info_gain (None, n_action)
            ids_score = regret_sq / info_gain                          # ids_score (None, n_action)
            q_values = -1. * ids_score
            # action = tf.argmin(ids_score, axis=-1)                   # (None,)
        elif action_selection == 'mean':
            action_values = self.q_network(obs, k=None)                # (1, 10, n_actions)
            q_values = tf.reduce_mean(action_values, axis=1)
        elif action_selection == 'sample':
            assert active_head is not None
            q_values = self.q_network(obs, k=active_head)                  # shape = (1, n_actions)
        else:
            raise Exception("action selection error.")

        deterministic_actions = tf.argmax(q_values, axis=1)                # compute Q-value and choose the best Q
        batch_size = tf.shape(obs)[0]
        # return a tensor with (batch_size,), each value [0, num_actions]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
        # return a tensor with (batch_size,), each value is True with p=eps
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
        # choose actions with epsilon-greedy
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        if stochastic:
            output_actions = stochastic_actions
        else:
            output_actions = deterministic_actions

        if update_eps >= 0:  # update_eps from LinearSchedule
            self.eps.assign(tf.convert_to_tensor(update_eps))
        return output_actions[0].numpy()

    def intrinsic_rew(self, states, actions, network, rew_type='none'):
        action_rank = tf.rank(actions).numpy()
        assert action_rank in [1, 2]                         # actions.shape = (None,) or (None, 10)
        assert rew_type in ['ucb', 'mcb']
        assert network in ['main', 'target']

        if network == 'main':                                       # intrinsic for immed-rew always use main-net
            action_values = self.q_network(states, k=None)          # (None, 10, n_actions)
        else:
            action_values = self.target_q_network(states, k=None)   # (None, 10, n_actions)

        # compute intrinsic reward
        if rew_type == 'ucb':
            inc_rew = tf.math.reduce_std(action_values, axis=1)            # (None, n_actions)
        elif rew_type == 'mcb':
            q_bar = tf.reduce_mean(action_values, axis=1, keepdims=True)   # (None, 1, n_actions)
            action_value_mcb = tf.maximum(action_values - q_bar, 0.)       # (None, 10, n_actions)
            inc_rew = tf.reduce_mean(action_value_mcb, axis=1)             # (None, n_actions)
        else:
            raise Exception("reward type error.")

        if action_rank == 1:     # for reward
            one_hot_action = tf.one_hot(actions, self.num_actions, dtype=tf.float32)  # (None, n_actions)
            inc_rew_action = tf.reduce_sum(inc_rew * one_hot_action, -1)              # (None, )
        else:                    # for next q
            one_hot_action = tf.one_hot(actions, self.num_actions, dtype=tf.float32)  # (None, 10, n_actions)
            inc_rew_action = tf.reduce_sum(tf.expand_dims(inc_rew, axis=1) * one_hot_action, -1)  # (None, 10)
        return inc_rew_action                      # (None, ) or (None, 10)

    def norm_intrinsic(self, rews_tf, norm_type='reward'):
        assert norm_type in ['reward', 'qvalue']
        rews = rews_tf.numpy()
        rffs_mean = np.mean(rews)
        rffs_std = np.std(rews)
        rffs_count = rews.ravel().shape[0]
        # print("mean-std-count:", rffs_mean, rffs_std, rffs_count)
        if norm_type == 'reward':
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            return rews_tf / np.sqrt(self.rff_rms.var)
        else:
            self.rff_rms_q.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            return rews_tf / np.sqrt(self.rff_rms_q.var)

    def compute_immed_reward(self, obs0, actions, reward_type, rew_immed_ratio):
        assert obs0.shape == (32, 84, 84, 4)
        normrew = self.normrew_ebu if self.ebu else self.normrew
        if reward_type == 'none' or (rew_immed_ratio - 0.0) < 1e-8:
            intrinsic_immediate_reward_norm = tf.zeros(obs0.shape.as_list()[0], dtype=tf.float32)
        else:
            intrinsic_immediate_reward = self.intrinsic_rew(obs0, actions=actions, rew_type=reward_type, network='main')  # (None, )
            # print("\ninc before norm:", intrinsic_immediate_reward.shape, intrinsic_immediate_reward.numpy().mean())
            if normrew:  # norm the reward
                intrinsic_immediate_reward_norm = self.norm_intrinsic(intrinsic_immediate_reward, norm_type='reward')
            else:
                intrinsic_immediate_reward_norm = intrinsic_immediate_reward
            # print("inc after norm:", intrinsic_immediate_reward_norm.numpy().mean())
        return intrinsic_immediate_reward_norm

    def compute_nextq_reward(self, obs1, next_action, reward_type, rew_nextq_ratio):
        normnxq = self.normnxq_ebu if self.ebu else self.normnxq
        # 2 -> add intrinsic reward to next-Q value
        if reward_type == 'none' or (rew_nextq_ratio - 0.0) < 1e-8:
            intrinsic_rew_target_q_norm = tf.zeros((obs1.shape.as_list()[0], self.num_ensemble), dtype=tf.float32)
        else:
            intrinsic_rew_target_q = self.intrinsic_rew(obs1, actions=next_action, rew_type=reward_type, network='target')  # (None, 10)
            # print("inc-q before norm:", intrinsic_rew_target_q.shape, intrinsic_rew_target_q.numpy().mean())
            if normnxq:  # norm
                intrinsic_rew_target_q_norm = self.norm_intrinsic(intrinsic_rew_target_q, norm_type='qvalue')
            else:
                intrinsic_rew_target_q_norm = intrinsic_rew_target_q
            # print("inc-q after norm:", intrinsic_rew_target_q_norm.shape, intrinsic_rew_target_q_norm.numpy().mean())
        return intrinsic_rew_target_q_norm

    def train_bdqn(self, replay_buffer, reward_type='none', rew_immed_ratio=0.001, rew_nextq_ratio=0.001):
        # sample.shape=(32, 84, 84, 4) (32,) (32,) (32, 84, 84, 4) (32,). dtype = uint8 int64 float32 uint8 float32
        obs0, actions, rewards, obs1, dones = replay_buffer.sample(self.batch_size)
        assert obs0.shape[1:] == obs1.shape[1:] == (84, 84, 4)
        assert reward_type in ['none', 'ucb', 'mcb']
        # 1 -> add intrinsic rewards to immediate reward
        # print("\n-----\n")
        intrinsic_immediate_reward_norm = self.compute_immed_reward(obs0, actions, reward_type, rew_immed_ratio)
        rewards = rewards + rew_immed_ratio * intrinsic_immediate_reward_norm  # (None, )

        with tf.GradientTape() as tape:
            # compute Q(s, a)
            q_t = self.q_network(obs0)                                                # (None, 10, num_actions)
            one_hot_action = tf.one_hot(actions, self.num_actions, dtype=tf.float32)  # (None, num_actions)
            q_t_selected = tf.einsum('bka,ba->bk', q_t, one_hot_action)               # (None, 10)
            # q_t_selected_2 = tf.reduce_sum(q_t * tf.expand_dims(one_hot_action, axis=1), -1)  # equal to q_t_selected

            # compute max_(a')[Q_target(s, a')] or Q(s, arg-max(Q_main(s, a)))
            q_tp1 = self.target_q_network(obs1)        # (None, 10, n_actions)
            next_action = tf.argmax(q_tp1, axis=-1)    # choose actions with argmax Q (None, 10)
            # print("next_action:", next_action.shape)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(next_action, self.num_actions, dtype=tf.float32), -1)  # (None, 10)
            # print("q_tp1_best:", q_tp1_best.shape)
            intrinsic_rew_target_q_norm = self.compute_nextq_reward(obs1, next_action, reward_type, rew_nextq_ratio)
            q_tp1_best = q_tp1_best + rew_nextq_ratio * intrinsic_rew_target_q_norm   # (None, 10)

            # target value
            dones = tf.expand_dims(tf.cast(dones, q_tp1_best.dtype), 1)  # (None, 1)
            q_tp1_best_masked = (1.0 - dones) * q_tp1_best  # (None, 10)
            q_t_selected_target = tf.expand_dims(rewards, 1) + self.gamma * q_tp1_best_masked  # (None, 10)

            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            if self.gradient_norm:
                loss = tf.reduce_mean(huber_loss(td_error)) / self.num_ensemble
            else:
                loss = tf.reduce_mean(huber_loss(td_error))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        grads_clip, _ = tf.clip_by_global_norm(grads, self.grad_norm_clipping)
        self.optimizer.apply_gradients(zip(grads_clip, self.q_network.trainable_variables))
        return loss.numpy()

    def train_bebu(self, replay_buffer, reward_type='none', rew_immed_ratio_ebu=0.001, rew_nextq_ratio_ebu=0.001):
        # print("self.batchnum:", self.batchnum, ", self.batch_count:", self.batch_count, ", stored:",
        #       self.epi_state.shape if self.epi_state is not None else 0)
        if self.batchnum == self.batch_count:
            self.epi_state, self.epi_actions, self.epi_rewards, self.batchnum, self.epi_terminals = replay_buffer.sample()  # sample a new episode
            self.epi_len = self.batchnum * self.batch_size
            # print("\n------\n sample s:", self.epi_state.shape, ", a:", len(self.epi_actions), ", r:", len(self.epi_rewards), ", d:", len(self.epi_terminals), ", batchnum=", self.batchnum)

            # intrinsic reward for immed reward
            intrinsic_immediate_reward_norm_list = [self.compute_immed_reward(self.epi_state[self.batch_size*i: self.batch_size*(i+1)],
                    tf.constant(self.epi_actions[self.batch_size*i: self.batch_size*(i+1)]), reward_type, rew_immed_ratio_ebu).numpy() for i in range(self.batchnum)]
            self.epi_rewards = self.epi_rewards + rew_immed_ratio_ebu * np.concatenate(intrinsic_immediate_reward_norm_list, axis=0)      # (None, )

            Q_tilde_list = [self.target_q_network(self.epi_state[self.batch_size * i:self.batch_size * (i + 1)]).numpy() for i in range(self.batchnum)]
            self.Q_tilde = np.concatenate(Q_tilde_list, axis=0)        # (None, 10, n_actions)
            self.Q_tilde = np.roll(self.Q_tilde, self.num_actions*self.num_ensemble)  # the first row become the second, and the last row becomes the first.

            if reward_type == 'none' or (rew_nextq_ratio_ebu - 0.0) < 1e-8:
                intrinsic_rew_target_q_norm = tf.zeros((self.epi_state.shape.as_list()[0], self.num_ensemble), dtype=tf.float32)
            else:   # intrinsic reward for next-q
                true_next_action = np.roll(self.epi_actions.copy(), 1)  # (None,)
                next_action = np.argmax(self.Q_tilde, axis=-1)  # choose actions with argmax Q (None, 10)
                next_state = np.roll(self.epi_state.numpy().copy(), 84 * 84 * 4)
                intrinsic_rew_target_q_norm_list = [self.compute_nextq_reward(
                    tf.constant(next_state[self.batch_size * i: self.batch_size * (i + 1)]),
                    tf.constant(next_action[self.batch_size * i: self.batch_size * (i + 1)]),
                    reward_type, rew_nextq_ratio_ebu).numpy() for i in range(self.batchnum)]
                bool_flag = np.stack([true_next_action for _ in range(self.num_ensemble)], axis=1) == next_action
                intrinsic_rew_target_q_norm = np.concatenate(intrinsic_rew_target_q_norm_list, axis=0) * (1.0 - bool_flag)  # (None, 10)
                intrinsic_rew_target_q_norm = rew_nextq_ratio_ebu * intrinsic_rew_target_q_norm  # (None, 10)

            for i in range(self.epi_len):
                if self.epi_terminals[i]:
                    self.Q_tilde[i] = 0.
            self.y_ = np.zeros((self.epi_len, self.num_ensemble), dtype=np.float32)  # (None, 10) Target value for each head.
            self.epi_rewards = np.expand_dims(self.epi_rewards, 1)    # (None, 1)
            for i in range(0, self.epi_len):
                if i < self.epi_len - 1:
                    #  The last minibatch stores some redundant transitions of the second episode to fill a minibatch,
                    #  so a terminal most likely occurs before self.epi_len
                    if self.epi_terminals[i]:
                        self.y_[i] = self.epi_rewards[i]
                        self.Q_tilde[i+1, :, self.epi_actions[i]] = self.beta * self.y_[i] + (1 - self.beta) * self.Q_tilde[i+1, :, self.epi_actions[i]]
                    elif self.epi_terminals[i + 1]:
                        self.y_[i] = self.epi_rewards[i] + self.gamma * (np.max(self.Q_tilde[i], axis=-1) + intrinsic_rew_target_q_norm[i])
                        self.Q_tilde[i+1, :] = 0.
                    else:
                        self.y_[i] = self.epi_rewards[i] + self.gamma * (np.max(self.Q_tilde[i], axis=-1) + intrinsic_rew_target_q_norm[i])
                        self.Q_tilde[i+1, :, self.epi_actions[i]] = self.beta * self.y_[i] + (1 - self.beta) * self.Q_tilde[i+1, :, self.epi_actions[i]]
                if i == self.epi_len - 1:                 # Most likely to be a transition of a redundant episode
                    if self.epi_terminals[i]:
                        self.y_[i] = self.epi_rewards[i]
                    else:
                        self.y_[i] = self.epi_rewards[i] + self.gamma * np.max(self.Q_tilde[i], axis=-1)

            self.batch_count = 1
            loss = self.train_bebu_step(
                        self.epi_state[0:self.batch_size],                      # state   (32, 84, 84, 4)
                        tf.constant(self.epi_actions[0:self.batch_size]),       # action  (32,)
                        tf.constant(self.y_[0:self.batch_size]))                # target  (32, 10)

        # if an episode is still being updated, use the next minibatch of the already generated target value.
        else:
            self.batch_count += 1
            # print("batch count:", self.batch_count)
            loss = self.train_bebu_step(
                self.epi_state[(self.batch_count - 1) * self.batch_size:self.batch_count * self.batch_size],
                tf.constant(self.epi_actions[(self.batch_count - 1) * self.batch_size: self.batch_count * self.batch_size]),
                tf.constant(self.y_[(self.batch_count - 1) * self.batch_size: self.batch_count * self.batch_size]))
        return loss

    def train_bebu_step(self, obs0, actions, y):
        # obs.shape = (None, 84, 84, 4),  actions.shape = (None,), y.shape = (None, 10)
        with tf.GradientTape() as tape:
            q_t = self.q_network(obs0)                                                # (None, 10, num_actions)
            one_hot_action = tf.one_hot(actions, self.num_actions, dtype=tf.float32)  # (None, num_actions)
            q_t_selected = tf.einsum('bka,ba->bk', q_t, one_hot_action)               # (None, 10)

            td_error = q_t_selected - tf.stop_gradient(y)
            if self.gradient_norm:
                loss = tf.reduce_mean(huber_loss(td_error)) / self.num_ensemble
            else:
                loss = tf.reduce_mean(huber_loss(td_error))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        grads_clip, grads_norm = tf.clip_by_global_norm(grads, self.grad_norm_clipping)
        self.optimizer.apply_gradients(zip(grads_clip, self.q_network.trainable_variables))
        return loss

    def update_target(self):
        if self.randomized_prior:
            self.target_q_network_main.set_weights(self.q_network_main.get_weights())
        else:
            self.target_q_network.set_weights(self.q_network.get_weights())


