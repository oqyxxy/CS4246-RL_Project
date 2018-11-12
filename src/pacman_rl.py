import sys
import gym
import operator
import warnings
import random
import numpy as np
from gym import wrappers
from PIL import Image
from collections import namedtuple

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, LeakyReLU
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, Memory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = "MsPacmanDeterministic-v4"
INPUT_SHAPE = (80, 80)  # Each frame will take 6400bytes of memory
WINDOW_LENGTH = 4


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high
        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick
        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation
    # Argument
        observation (list): List of observation
    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class SegmentTree(object):
    """
    Abstract SegmentTree data structure used to create PrioritizedMemory.
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity, operation, neutral_element):

        #powers of two have no bits in common with the previous integer
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2"
        self._capacity = capacity

        #a segment tree has (2*n)-1 total nodes
        self._value = [neutral_element for _ in range(2 * capacity)]

        self._operation = operation

        self.next_index = 0

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """
    SumTree allows us to sum priorities of transitions in order to assign each a probability of being sampled.
    """
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    """
    In PrioritizedMemory, we normalize importance weights according to the maximum weight in the buffer.
    This is determined by the minimum transition priority. This MinTree provides an efficient way to
    calculate that.
    """
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class PartitionedRingBuffer(object):
    """
    Buffer with a section that can be sampled from but never overwritten.
    Used for demonstration data (DQfD). Can be used without a partition,
    where it would function as a fixed-idxs variant of RingBuffer.
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.length = 0
        self.data = [None for _ in range(maxlen)]
        self.permanent_idx = 0
        self.next_idx = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            raise KeyError()
        return self.data[idx % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        self.data[(self.permanent_idx + self.next_idx)] = v
        self.next_idx = (self.next_idx + 1) % (self.maxlen - self.permanent_idx)

    def load(self, load_data):
        assert len(load_data) < self.maxlen, "Must leave space to write new data."
        for idx, data in enumerate(load_data):
            self.length += 1
            self.data[idx] = data
            self.permanent_idx += 1


class PERMemory(Memory):
    def __init__(self, limit, alpha=.4, start_beta=1., end_beta=1., steps_annealed=1, **kwargs):
        super(PERMemory, self).__init__(**kwargs)

        #The capacity of the replay buffer
        self.limit = limit

        #Transitions are stored in individual RingBuffers, similar to the SequentialMemory.
        self.actions = PartitionedRingBuffer(limit)
        self.rewards = PartitionedRingBuffer(limit)
        self.terminals = PartitionedRingBuffer(limit)
        self.observations = PartitionedRingBuffer(limit)

        assert alpha >= 0
        #how aggressively to sample based on TD error
        self.alpha = alpha
        #how aggressively to compensate for that sampling. This value is typically annealed
        #to stabilize training as the model converges (beta of 1.0 fully compensates for TD-prioritized sampling).
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.steps_annealed = steps_annealed

        #SegmentTrees need a leaf count that is a power of 2
        tree_capacity = 1
        while tree_capacity < self.limit:
            tree_capacity *= 2

        #Create SegmentTrees with this capacity
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.

        #wrapping index for interacting with the trees
        self.next_index = 0

    def append(self, observation, action, reward, terminal, training=True):\
        #super() call adds to the deques that hold the most recent info, which is fed to the agent
        #on agent.forward()
        super(PERMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            #The priority of each new transition is set to the maximum
            self.sum_tree[self.next_index] = self.max_priority ** self.alpha
            self.min_tree[self.next_index] = self.max_priority ** self.alpha

            #shift tree pointer index to keep it in sync with RingBuffers
            self.next_index = (self.next_index + 1) % self.limit

    def _sample_proportional(self, batch_size):
        #outputs a list of idxs to sample, based on their priorities.
        idxs = list()

        for _ in range(batch_size):
            mass = random.random() * self.sum_tree.sum(0, self.limit - 1)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idxs.append(idx)

        return idxs

    def sample(self, batch_size, beta=1., nstep=1, gamma=1.):
        idxs = self._sample_proportional(batch_size)

        #importance sampling weights are a stability measure
        importance_weights = list()

        #The lowest-priority experience defines the maximum importance sampling weight
        prob_min = self.min_tree.min() / self.sum_tree.sum()
        max_importance_weight = (prob_min * self.nb_entries)  ** (-beta)
        obs_t, act_t, rews, obs_t1, dones = [], [], [], [], []

        experiences = list()
        for idx in idxs:
            while idx < self.window_length + 1:
                idx += 1
            while idx + nstep > self.nb_entries and self.nb_entries < self.limit:
                # We are fine with nstep spilling back to the beginning of the buffer
                # once it has been filled.
                idx -= 1
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries - nstep, size=1)[0]
                terminal0 = self.terminals[idx - 2]

            assert self.window_length + 1 <= idx < self.nb_entries

            #probability of sampling transition is the priority of the transition over the sum of all priorities
            prob_sample = self.sum_tree[idx] / self.sum_tree.sum()
            importance_weight = (prob_sample * self.nb_entries) ** (-beta)
            #normalize weights according to the maximum value
            importance_weights.append(importance_weight/max_importance_weight)

            #assemble the initial state from the ringbuffer.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))

            action = self.actions[idx - 1]
            # N-step TD
            reward = 0
            nstep = nstep
            for i in range(nstep):
                reward += (gamma**i) * self.rewards[idx + i - 1]
                if self.terminals[idx + i - 1]:
                    #episode terminated before length of n-step rollout.
                    nstep = i
                    break

            terminal1 = self.terminals[idx + nstep - 1]

            # We assemble the second state in a similar way.
            state1 = [self.observations[idx + nstep - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx + nstep - 1 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state1.insert(0, self.observations[current_idx])
            while len(state1) < self.window_length:
                state1.insert(0, zeroed_observation(state0[0]))

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size

        # Return a tuple whre the first batch_size items are the transititions
        # while -2 is the importance weights of those transitions and -1 is
        # the idxs of the buffer (so that we can update priorities later)
        return tuple(list(experiences)+ [importance_weights, idxs])

    def update_priorities(self, idxs, priorities):
        #adjust priorities based on new TD error
        for i, idx in enumerate(idxs):
            assert 0 <= idx < self.limit
            priority = priorities[i] ** self.alpha
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def calculate_beta(self, current_step):
        a = float(self.end_beta - self.start_beta) / float(self.steps_annealed)
        b = float(self.start_beta)
        current_beta = min(self.end_beta, a * float(current_step) + b)
        return current_beta

    def get_config(self):
        config = super(PERMemory, self).get_config()
        config['alpha'] = self.alpha
        config['start_beta'] = self.start_beta
        config['end_beta'] = self.end_beta
        config['beta_steps_annealed'] = self.steps_annealed

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)


class AtariProcessor(Processor):
    def process_observation(self, observation):
        # We perform standard atari pre-processing here to optimize training performance.
        # Optimization includes cropping, resizing and gray-scaling.
        # Use "img.show()" to view how the resulting image looks like.
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.crop((0, 0, 160, 172))  # Removes bottom interface menu (Lives, score, cherries)
        img = img.resize(INPUT_SHAPE).convert('L')  # Resize + gray-scale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    # def process_reward(self, reward):
    #     return np.sign(reward)
    #
    # def process_info(self, info):
    #     if info['ale.lives'] < 1:
    #         print(info['ale.lives'])
    #     return info


def build_nn_model():
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 8, strides=4))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, strides=2))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


# Basic input validation
if len(sys.argv) < 2:
    print("Invalid input format! Expected format:")
    print("python pacman_rl.py <train/run>")


execution_mode = sys.argv[1]
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n
# env = wrappers.Monitor(env, './', force=True)  # save animations


# Build our model with reference to deepmind DQN paper (Minh et al.).
model = build_nn_model()
print(model.summary())


# memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
memory = PERMemory(limit=1000000, alpha=.6, start_beta=.4, end_beta=1., steps_annealed=30000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1., enable_double_dqn=True, enable_dueling_network=True)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if execution_mode == 'train':
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=10000000, log_interval=1000000)
    dqn.save_weights(weights_filename, overwrite=True)
    dqn.test(env, nb_episodes=10, visualize=False)
elif execution_mode == 'run':
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
