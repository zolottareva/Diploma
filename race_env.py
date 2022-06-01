from collections import deque

import gym.utils
from gym import spaces
import numpy as np
import cv2

from track_generator import generate_track
from utils import Border

class RaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config=None):
        config = config or {}
        render_mode = config.get('render_mode')
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        car_x, car_y = self.car_position = np.array([0, 0], float)
        self.car_size = np.array([10, 10])
        self.position_history = deque(maxlen=25)
        self.velocity = 10 # V
        self.direction = 0 # Theta
        self.rays_count = 25
        self.ray_max_distance = 450
        self.vision_range = [-45, 45]
        self.frames_count = 0
        self.steps_count = 0
        # self.borders = [
        #     Border(car_x, car_y + 30, car_x + 1000, car_y + 30),
        #     Border(car_x, car_y - 30, car_x + 1000, car_y - 30),
        #     Border(car_x + 1000, car_y - 30, car_x + 1500, car_y + 300),
        #     Border(car_x + 1000, car_y + 30, car_x + 1420, car_y + 300)
        # ]
        self.borders, self.finish_line = generate_track()
        self.rays = []
        self.window_size = 1024  # The size of the PyGame window
        self.max_velocity_change = 1
        self.min_velocity = 5
        self.max_velocity = 100
        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(0, 1, shape=(self.rays_count,), dtype=float),
                "velocity": spaces.Box(self.min_velocity / self.max_velocity, 1, shape=(1,), dtype=float)
            }
        )
        self.turn_limit = 1
        self.action_space = spaces.Dict({
            'angle_change': spaces.Box(low=-self.turn_limit, high=self.turn_limit, shape=[1], dtype=float),
            'velocity_change': spaces.Box(low=-self.max_velocity_change, high=self.max_velocity_change, shape=[1], dtype=float),
        })
        self.window = None
        if render_mode == "human":
            import pygame
            pygame.font.init()
            self.font = pygame.font.SysFont('Comic Sans MS', 30)
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
    
    def _get_ray_collision_distance(self, ray):
        minimal = self.ray_max_distance
        for b in self.borders:
            if b.is_crossing(ray):
                point = b.get_crossing_point(ray)
                if point is not None:
                    distance = np.linalg.norm(self.car_position - point)
                    minimal = min(distance, minimal)
        return minimal

    def _get_obs(self):
        vision = [self._get_ray_collision_distance(r) / self.ray_max_distance for r in self.rays]
        return {'vision' : vision, 'velocity': [self.velocity / self.max_velocity]}
    
    def _get_info(self):
        return {}

    def _get_car_cords(self):
        return np.array([
                [self.car_position[0] - self.car_size[0]/2, self.car_position[1] + self.car_size[1]/2], # top left
                [self.car_position[0] + self.car_size[0]/2, self.car_position[1] + self.car_size[1]/2], # top right
                [self.car_position[0] + self.car_size[0]/2, self.car_position[1] - self.car_size[1]/2], # bottom right
                [self.car_position[0] - self.car_size[0]/2, self.car_position[1] - self.car_size[1]/2]  # bottom left
        ])
    
    def _update_rays(self):
        self.rays.clear()
        for angle in np.linspace(*self.vision_range, self.rays_count, endpoint=True):
            global_angle = self.direction + angle
            ray = Border.from_point_angle(self.car_position, global_angle, self.ray_max_distance)
            self.rays.append(ray)

    def _get_car_borders(self):
        cords = self._get_car_cords()
        borders = []
        for i, (x, y) in enumerate(cords):
            next_vert = cords[(i + 1) % 4]
            borders.append(Border(x, y, *next_vert))
        return borders
    
    def is_collided(self):
        car_borders = self._get_car_borders()
        for b in self.borders:
            for c_b in car_borders:
                if b.is_crossing(c_b):
                    return True
        return False
            
    def is_finished(self):
        return any(b.is_crossing(self.finish_line) for b in self._get_car_borders())
    
    def step(self, action):
        angle_change, velocity_change = action['angle_change'], action['velocity_change']
        self.steps_count += 1
        self.direction += angle_change[0]
        self.velocity = np.clip(self.velocity + velocity_change[0], self.min_velocity, self.max_velocity)
        direction = np.deg2rad(self.direction)
        delta_x = self.velocity * np.cos(direction)
        delta_y = self.velocity * np.sin(direction)
        
        self.car_position += [delta_x, delta_y]
        self.position_history.append(self.car_position.copy())

        self._update_rays()

        if self.is_collided() or self.steps_count > 1000:
            reward = 0
            done = True
        elif self.is_finished():
            reward = 100 - self.steps_count
            done = True
        else:
            reward = 0.001
            done = False
        obs = self._get_obs()
        if self.render_mode is not None:
            self._render_frame(obs)
        
        return obs, reward, done, self._get_info()
        
    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self, obs):
        assert self.render_mode is not None
        import pygame
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        window_center = self.car_position - [self.window_size / 2] * 2

        # Draw the car
        car_borders = self._get_car_borders()
        for b in car_borders:
            pygame.draw.line(
                canvas,
                0,
                *(b.points() - window_center),
                width=3,
            )
        
        for i in range(len(self.position_history) - 1):
            pygame.draw.line(
                canvas,
                (0, 100, 0),
                self.position_history[i] - window_center,
                self.position_history[i + 1] - window_center,
                width=3,
            )

        # Draw the track
        for b in self.borders:
            pygame.draw.line(
                canvas,
                (255, 0, 255),
                *(b.points() - window_center),
                width=3,
            )
        
        for b, intense in zip(self.rays, obs['vision']):
            pygame.draw.line(
                canvas,
                np.array([255, 100, 50]) * intense,
                *(b.points() - window_center),
                width=3,
            )

        # Draw finish line
        pygame.draw.line(
            canvas,
            (255, 0, 0),
            *(self.finish_line.points() - window_center),
            width=3,
        )

        text_surface = self.font.render(str(self.steps_count), False, (0, 0, 0))
        canvas.blit(text_surface, (0, 0))
        text_surface = self.font.render(f'{self.velocity:.1f}', False, (100, 0, 0))
        canvas.blit(text_surface, (100, 0))
        if self.render_mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            cv2.imwrite(f"frames/{self.frames_count}.jpg", frame)
            self.frames_count += 1


    def reset(self, seed=None, return_info=False, options=None):
        # Choose the agent's location uniformly at random
        self.direction = 0
        self.velocity = 10
        self.steps_count = 0
        self.car_position.fill(0)
        self.position_history.clear()
        self.borders, self.finish_line = generate_track()
        # clean the render collection and add the initial frame
        self._update_rays()
        observation = self._get_obs()
        if self.render_mode is not None:
            self._render_frame(observation)

        info = self._get_info()
        return (observation, info) if return_info else observation
