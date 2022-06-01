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
        action = trainer.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

for i in range(100):
    print(i, trainer.train()['episode_reward_mean'])
    path = trainer.save(f'checkpoints/')
    if i % 10 == 0:
        run_human_evaluation()
input('Ready?')
run_human_evaluation()

