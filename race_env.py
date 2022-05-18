import gym.utils
from gym import spaces
import numpy as np
from typing import Optional

class Border:
    def __init__(self, x1, y1, x2, y2):
        self.line = np.array([x1, y1, x2, y2])
    
    def is_crossing(self, border):
        k_self, b_self = self.count_k_b()
        k_border, b_border = border.count_k_b()
        if k_self == k_border:
            return False # parallel
        x_crossing = (b_self - b_border)/(b_border - b_self)
        y_crossing = k_self * x_crossing + b_self
        return self.check_if_inside(x_crossing, y_crossing)
        
    def count_k_b(self):
        k = (self.line[3] - self.line[1])/(self.line[2]-self.line[0]) # (y2-y1)/(x2-x1)
        b = self.line[1] - k * self.line[0] # y1 - k * x1
        return (k, b)

    def _check_if_inside(self, x, y):
        if self.line[0] != self.line[2]:
            return x >= min(self.line[0], self.line[2]) and x <= max(self.line[0], self.line[2])
        return y >= min(self.line[1], self.line[3]) and y <= max(self.line[1], self.line[3])
    

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

    def _get_car_cords(self):
        return np.array([
                [self.car_position[0] - self.car_size[0]/2, self.car_position[1] + self.car_size[1]/2], # top left
                [self.car_position[0] + self.car_size[0]/2, self.car_position[1] + self.car_size[1]/2], # top right
                [self.car_position[0] + self.car_size[0]/2, self.car_position[1] - self.car_size[1]/2], # bottom right
                [self.car_position[0] - self.car_size[0]/2, self.car_position[1] - self.car_size[1]/2]  # bottom left
        ])
    
    def _get_car_borders(self):
        cords = self._get_car_cords()
        borders = []
        for i, (x, y) in enumerate(cords):
            next_vert = cords[(i + 1) % 4]
            borders.append(Border(x, y, next_vert[0], next_vert[1]))
    
    def is_colided(self):
        car_borders = self._get_car_borders()
        for b in self.borders:
            for c_b in car_borders:
                if b.is_crossing(c_b):
                    return True
        return False
            
    def is_finished(self):
        for b in self.borders:
            if b.is_crossing(self.finish_line):
                return True
        return False
    
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
        
        