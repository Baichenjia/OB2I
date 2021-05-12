import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class ReplayBuffer(object):
    def __init__(self, size, frame_height=84, frame_width=84):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self._maxsize = size
        self._next_idx = 0    # index to add example
        self._storage = 0     # the current storage
        self.obs_len = 4      # frame stack = 4

        # Pre-allocate memory
        self.states = np.empty((size, frame_height, frame_width), dtype=np.uint8)
        self.actions = np.empty(size, dtype=np.int64)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.empty(size, dtype=np.bool)

    def add(self, obs_t, action, reward, done):
        # obs_t.shape == (84, 84, 4), action is scalar, reward is scalar, done in [True, False]
        self.states[self._next_idx] = obs_t[:, :, -1]  # choose the last state of frame_stck
        self.actions[self._next_idx] = action
        self.rewards[self._next_idx] = reward
        self.dones[self._next_idx] = done

        # update index and storage
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._storage = min(self._maxsize, self._storage + 1)

    def _encode_img_observation(self, idx):
        """ Encode the observation for idx by stacking the `4` preceding frames together.
            Assume there are more than `obs_len` frames in the buffer.
        """
        hi = idx + 1  # make noninclusive
        lo = hi - self.obs_len

        for i in range(lo, hi - 1):
            if self.dones[i % self._maxsize]:
                lo = i + 1
        missing = self.obs_len - (hi - lo)

        # We need to duplicate the lo observation
        if missing > 0:
            frames = [self.states[lo % self._maxsize] for _ in range(missing)]
            for i in range(lo, hi):
                frames.append(self.states[i % self._maxsize])
            stack_state = np.stack(frames, axis=-1)
        # We are on the boundary of the buffer
        elif lo < 0:
            frames = [self.states[lo:], self.states[:hi]]
            frames = np.concatenate(frames, 0)
            stack_state = frames.transpose((1, 2, 0))
        # The standard case
        else:
            stack_state = self.states[lo:hi].transpose((1, 2, 0))

        assert stack_state.shape == (84, 84, 4)
        return stack_state

    def _exclude_indices(self):
        """Compute indices that must be excluded because the information there might be inconsistent
        """
        idx = self._next_idx
        exclude = np.arange(idx - 1, idx + self.obs_len) % self._maxsize
        return exclude

    @staticmethod
    def _sample_n_unique(n, lo, hi, exclude=None):
        """Sample n unique indices in the range [lo, hi), making sure no sample appreas in `exclude`
        Args:
            n: int. Number of samples to take
            lo: int. Lower boundary of the sample range; inclusive
            hi: int. Upper boundary of the sample range; exclusive
            exclude: list or np.array. Contains values that samples must not take
        Returns:
            np.array of the sampled indices
        """
        batch = np.empty(n, dtype=np.uint32)
        k = 0
        while k < n:
            samples = np.random.randint(low=lo, high=hi, size=n - k)
            samples = np.unique(samples)  # Get only the unique entries
            # Get only the entries which are not in exclude
            if exclude is not None:
                valid = np.all(samples[:, None] != exclude, axis=-1)
                # print("***", (samples[:, None] != exclude).shape, valid)   # （32, 5）
                samples = samples[valid]  # (None,) contains True or False
                # print("samples:", samples)
            # Update batch
            end = min(k + samples.shape[0], n)
            batch[k:end] = samples
            k = end
        return batch

    def sample(self, batch_size):
        exclude = self._exclude_indices()
        assert batch_size < self._storage - len(exclude) - 1
        inds = self._sample_n_unique(batch_size, 0, self._storage, exclude)
        next_inds = (inds + 1) % self._maxsize

        obs_batch = np.concatenate([self._encode_img_observation(idx)[None] for idx in inds], 0)
        obs_tp1_batch = np.concatenate([self._encode_img_observation(idx)[None] for idx in next_inds], 0)

        act_batch = self.actions[inds]
        rew_batch = self.rewards[inds]
        done_batch = self.dones[inds].astype(np.float32)
        return tf.constant(obs_batch), tf.constant(act_batch), tf.constant(rew_batch), \
               tf.constant(obs_tp1_batch), tf.constant(done_batch)


class ReplayBufferEBU(object):
    def __init__(self, size, frame_height=84, frame_width=84, batch_size=32):
        """ Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self._maxsize = size
        self._next_idx = 0        # Add samples in replay buffer
        self._storage = 0         # the current storage
        self.batch_size = batch_size

        # Pre-allocate memory
        self.states = np.empty((size, frame_height, frame_width), dtype=np.uint8)
        self.actions = np.empty(size, dtype=np.int64)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool)

    def add(self, obs_t, action, reward, done):
        self.states[self._next_idx] = obs_t[:, :, -1]
        self.actions[self._next_idx] = action
        self.rewards[self._next_idx] = reward
        self.dones[self._next_idx] = done

        # update index and storage
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._storage += 1
        self._storage = min(self._maxsize, self._storage)

    def find_done_index(self):
        # find the position of done. We must delete the transitions after self._next_idx because this episode is not complete.
        terminal_array = np.where(self.dones == True)[0]
        if self._storage == self._maxsize:
            for j in range(self._next_idx, self._maxsize):
                if self.dones[j] == True:
                    delete_index = j
                    # print("terminal_array:", terminal_array)
                    # print("delete_index:", delete_index)
                    return np.delete(terminal_array, np.where(terminal_array == delete_index))
                else:
                    return terminal_array
        else:
            return terminal_array
        return terminal_array

    def sample(self):
        terminal_array = self.find_done_index()

        # batchnum = 0
        # while batchnum == 0:
        # exclude some early and final episodes from sampling due to indexing issues,
        # sample two episodes (ind1 for main, and ind2 for the remaining steps to make multiple of 32)
        ind = np.random.choice(range(5, len(terminal_array)-3), 2, replace=False)
        ind1 = ind[0]
        ind2 = ind[1]

        indice_array = range(terminal_array[ind1], terminal_array[ind1-1], -1)    # reverse manner
        epi_len = len(indice_array)
        batchnum = int(np.ceil(epi_len/float(self.batch_size)))                   # upper 
        assert batchnum > 0

        remainindex = int(batchnum * self.batch_size + 3 - epi_len)
        # print("remainindex:", remainindex, ", first episode length:", epi_len, ", ind1:", ind1, ", ind2:", ind2)

        # Normally an episode does not have steps=multiple of 32.
        # Fill last minibatch with redundant steps from another episode
        indice_array = np.append(indice_array, range(terminal_array[ind2], terminal_array[ind2]-remainindex, -1))
        indice_array = indice_array.astype(int)
        # print("sample index:", indice_array, ", length:", indice_array.shape)

        # SAMPLE
        dones = self.dones[indice_array]
        states = self.states[indice_array].copy()                 # (None,84,84)

        # print(dones.shape, dones.astype(np.int))
        # print(states.shape)
        # states
        states_stack_list = []
        for s_idx in range(0, states.shape[0]-3):
            if dones[s_idx + 1] == 1:
                s_stack = states[np.array([s_idx, s_idx, s_idx, s_idx])]
            elif dones[s_idx + 2] == 1:
                s_stack = states[np.array([s_idx+1, s_idx+1, s_idx+1, s_idx])]
            elif dones[s_idx + 3] == 1:
                s_stack = states[np.array([s_idx+2, s_idx+2, s_idx+1, s_idx])]
            else:
                s_stack = states[np.array([s_idx+3, s_idx+2, s_idx+1, s_idx])]
            states_stack_list.append(s_stack)
        states_stack = np.stack(states_stack_list, axis=0).transpose((0, 2, 3, 1))  # (None,84,84,4)
        # print(states_stack.shape)

        rewards = self.rewards[indice_array]
        actions = self.actions[indice_array]
        return tf.constant(states_stack), actions[:-3], rewards[:-3], batchnum, dones[:-3]

