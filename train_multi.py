
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune.registry import register_env

from race_env_multi import MultiAgentRaceEnv

car_numbers = 4
env_config = {'turns_count': 5, 'cars_number': car_numbers}
check_env(MultiAgentRaceEnv(env_config))

register_env(
    "race_multi",
    lambda _: (MultiAgentRaceEnv(env_config)),
)


policies = {
    str(i): PolicySpec()
    for i in range(car_numbers)
}
config = (
    PPOConfig()
    .rollouts(
        num_rollout_workers=12,
        num_envs_per_worker=1,
        batch_mode='complete_episodes',
        sample_async=True
    )
    .fault_tolerance()
    .framework('torch')
    .training(gamma=0.99, lr=0.00001, entropy_coeff=0.001)
    .environment(env='race_multi')
    .multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, *_, **__: agent_id
    )
)
trainer = config.build()

# trainer.restore('checkpoints_with_competitors_multi_policy/checkpoint_000161')
reward_history = []
for i in range(1000):
    epoch_info = trainer.train()
    episode_reward = epoch_info['episode_reward_mean']
    episodes_lengths = epoch_info['hist_stats']['episode_lengths']
    print(i, episode_reward, (epoch_info['num_env_steps_trained_this_iter']))
    reward_history.append(episode_reward)
    if i % 20 == 0:
        #run_human_evaluation()
        path = trainer.save('checkpoints_with_competitors_multi_policy/')
        plt.plot(reward_history)
        plt.savefig('res.png')


def run_human_evaluation():
    env = MultiAgentRaceEnv({"render_mode": "human", **env_config})

    episode_reward = 0

    obs, info = env.reset()
    for step in range(500):
        actions = {}
        for agent_id, ob in obs.items():
            policy_id = trainer.config['multiagent']['policy_mapping_fn'](agent_id)
            actions[agent_id] = trainer.compute_single_action(ob, policy_id=policy_id)
        obs, rewards, dones, trunc, infos = env.step(actions)
        episode_reward += sum(rewards.values())

        if dones['__all__']:
            break
    print(f'{step=}')
print('starting')
run_human_evaluation()
print('finished')
input()
