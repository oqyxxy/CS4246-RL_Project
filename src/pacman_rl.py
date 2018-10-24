import sys
import gym
import numpy as np
from gym import wrappers
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, LeakyReLU
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = "MsPacmanDeterministic-v4"
INPUT_SHAPE = (80, 80)  # Each frame will take 6400bytes of memory
WINDOW_LENGTH = 4


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


memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
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
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=10000000, log_interval=1000000)
    dqn.save_weights(weights_filename, overwrite=True)
    dqn.test(env, nb_episodes=10, visualize=False)
elif execution_mode == 'run':
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
