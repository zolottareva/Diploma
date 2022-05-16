import gym.utils
from gym import spaces
import numpy as np
from typing import Optional

class Border:
    def __init__(self, x1, y1, x2, y2) -> None:
        self.line = np.array([x1, y1, x2, y2])

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        car_x, car_y = self.car_position = np.array([0, 0])
        self.finish_line = Border(car_x + 100, car_y + 30, car_x + 100, car_y - 30)
        self.car_size = np.array([10, 10])
        self.velocity = 10 # V
        self.direction = 0 # Theta
        
        self.borders = [
            Border(car_x, car_y + 30, car_x + 100, car_y + 30),
            Border(car_x, car_y - 30, car_x + 100, car_y - 30),
        ]
        
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(0, 1, shape=(5,), dtype=float)
            }
        )

        self.turn_limit = 0.5
        self.action_space = spaces.Box(-self.turn_limit, self.turn_limit, dtype=float)

        if render_mode == "human":
            # TODO: vizualize
            ...
            # pygame.init()
            # pygame.display.init()
            # self.window = pygame.display.set_mode((self.window_size, self.window_size))
            # self.clock = pygame.time.Clock()
    
    def _get_obs(self):
        # TODO: measure how far the obstacles are
        return {"vision": []}
    
    def is_colided(self):
        # TODO: check for collision    
        ...
    
    def is_finished(self):
        # TODO: check if car has crossed finish line
        ...
    
    def step(self, action):
        self.direction += action
        delta_x = self.velocity * np.cos(self.direction)
        delta_y = self.velocity * np.sin(self.direction)
        
        self.car_position += [delta_x, delta_y]
        
        if self.is_colided():
            reward = 0
            done = True
        elif self.is_finished():
            reward = 1
            done = True
        else:
            reward = 0.5
            done = False
        
        frame = self.render_step()
        
        return self._get_obs(), reward, done, None
        
        