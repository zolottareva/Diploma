from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from race_env_multi import MultiAgentRaceEnv
from ray.tune.registry import register_env
car_numbers = 4
env_config = {'turns_count': 5, 'cars_number': car_numbers, "render_mode": "human"}

load_env = lambda _: (MultiAgentRaceEnv(env_config))
register_env(
    "race_multi",
    load_env,
)
policies = {
    str(i): PolicySpec()
    for i in range(car_numbers)
}
trainer = (
    PPOConfig()
    .framework('torch')
    .environment('race_multi')
    .multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, *_, **__: '0'
    )
    .build()
)
trainer.restore('checkpoints_with_competitors_multi_policy/checkpoint_000161')
def run_human_evaluation():
    episode_reward = 0
    env = load_env(None)
    obs, info = env.reset()
    for step in range(5000):
        actions = {}
        for agent_id, ob in obs.items():
            env.action_space.sample()
            policy_id = trainer.config['multiagent']['policy_mapping_fn'](agent_id, episode=None, worker=None)
            actions[agent_id] = trainer.compute_single_action(ob, policy_id=policy_id, explore=False)
        obs, rewards, dones, trunc, infos = env.step(actions)
        episode_reward += sum(rewards.values())

        if dones['__all__']:
            break
run_human_evaluation()
