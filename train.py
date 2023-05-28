from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddppo import DDPPOConfig
from ray.rllib.algorithms import Algorithm
from race_env import RaceEnv
from gymnasium.wrappers import TimeLimit
from ray.tune.registry import register_env

import matplotlib.pyplot as plt


register_env(
    "race_truncated",
    lambda _: TimeLimit(RaceEnv({'turns_count': 10}), max_episode_steps=150),
)
config = (PPOConfig()
        .rollouts(num_rollout_workers=12, num_envs_per_worker=2)
        .framework('torch')
        .training(gamma=0.9, lr=0.001, train_batch_size=1000)
        .environment(env='race_truncated')
        .exploration())

trainer = (
    config
    .build()
)
if 1:
    trainer.restore('checkpoints/checkpoint_003041')
def run_human_evaluation():
    env = RaceEnv({"render_mode": "human"})

    episode_reward = 0
    done = False
    obs, info = env.reset()
    for _ in range(500):
        action = trainer.compute_single_action(obs, explore=False)
        obs, reward, done, trunc, info = env.step(action)
        episode_reward += reward
        if done:
            break
reward_history = []
for i in range(10000):
    episode_reward = trainer.train()['episode_reward_mean']
    print(i, episode_reward)
    reward_history.append(episode_reward)
    if i % 20 == 0:
        #run_human_evaluation()
        path = trainer.save(f'checkpoints/')
        plt.plot(reward_history)
        plt.savefig('res.png')

