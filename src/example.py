import gym


ENV_NAME = 'MsPacman-ram-v0'


env = gym.make(ENV_NAME)
nb_actions = env.action_space.n



for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break