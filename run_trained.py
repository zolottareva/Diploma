from ray.rllib.agents.ppo import PPOTrainer
from race_env import RaceEnv

config = {
    "framework": "torch",
    'env_config': {"render_mode": None}
}

trainer = PPOTrainer(env=RaceEnv, config=config)
def run_human_evaluation():
    env = RaceEnv({"render_mode": "human",})

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = trainer.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
trainer.restore('checkpoints/checkpoint_000100/checkpoint-100')
run_human_evaluation()