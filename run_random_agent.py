import gym
from race_env import RaceEnv
env = RaceEnv({'render_mode':"rgb_array"})

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())

    if done:
        observation, info = env.reset(return_info=True)

env.close()
