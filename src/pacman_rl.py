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
INPUT_SHAPE = (80, 80)
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
    #     if reward > 10:
    #         print(reward)
    #     elif reward < 0:
    #         print(reward)
    #     return np.clip(reward, -1., 1.)


def build_nn_model():
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 8, strides=4))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, strides=2))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.3))
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
# env = wrappers.Monitor(env, './', force=True)  # save animations
nb_actions = env.action_space.n


# Build our model with reference to deepmind DQN paper (Minh et al.).
model = build_nn_model()
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1., enable_double_dqn=True, enable_dueling_network=True)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if execution_mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=10000000, log_interval=1000000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif execution_mode == 'run':
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    # if args.weights:
    #     weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
