from collections import deque
import random

from gymnasium import spaces
import numpy as np
import cv2
from ray.rllib import MultiAgentEnv

from track_generator import generate_track
from utils import Border

class MultiAgentRaceEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config={}):
        super().__init__()
        config: dict = config.copy()
        config.setdefault('cars_number', 2)
        config.setdefault('turns_count', 10)
        config.setdefault('render_mode')
        self.config = config
        render_mode = config['render_mode']
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        self.cars_number = config['cars_number']
        
        self.positions: dict = None
        self.directions: dict = None
        self.velocities: dict = None
        self.rays_list: dict = None
        self.dones: dict = None
        self.cumulative_rewards: dict = None
        self.steps_count = 0
        
        self.car_size = np.array([10, 10])
        self.position_history = {i: deque(maxlen=25) for i in self.iterate_agents()}
        self.rays_count = 25
        self.ray_max_distance = 200
        self.vision_range = [-90, 90]
        self.frames_count = 0
        
        self.max_episode_steps = 1000
        
        self.render_mode = render_mode
        
        self.borders, self.finish_line, self.turns = [None] * 3
        self.window_size = 1024  # The size of the PyGame window
        self.max_velocity_change = 1
        self.min_velocity = 1
        self.max_velocity = 10
        
        self.max_recorded_competitor_distance = self.ray_max_distance
        
        agent_observation_space = spaces.Dict(
            {
                "vision": spaces.Box(0, 1, shape=(self.rays_count,), dtype=float),
                "velocity": spaces.Box(self.min_velocity / self.max_velocity, 1, shape=(1,), dtype=float),
                "turn_angle": spaces.Box(-1, 1, shape=(2,), dtype=float),
                "competitor_distances": spaces.Box(
                    0, self.max_recorded_competitor_distance, 
                    shape=(self.cars_number - 1,),
                    dtype=float
                ),
                "competitor_angles": spaces.Box(
                    -1, 1,
                    shape=(self.cars_number - 1, 2),
                    dtype=float
                )
            }
        )
        self.observation_space = agent_observation_space
        self.turn_limit = 1
        agent_action_space = spaces.Dict({
            'angle_change': spaces.Box(low=-self.turn_limit, high=self.turn_limit, shape=[1], dtype=float),
            'velocity_change': spaces.Box(low=-self.max_velocity_change, high=self.max_velocity_change, shape=[1], dtype=float),
        })
        
        self.action_space = agent_action_space
        
        
        self.window = None
        if render_mode == "human":
            import pygame
            pygame.font.init()
            self.font = pygame.font.SysFont('Comic Sans MS', 30)
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
    
    def generate_start_positions(self):
        positions = [np.array([i * 20, 0.0]) for i, _ in enumerate(self.iterate_agents())]
        #random.shuffle(positions)
        return {i: p for i, p in zip(self.iterate_agents(), positions)}
    
    def iterate_agents(self):
        return (str(i) for i in range(self.cars_number))
    
    def _get_ray_collision_distance(self, ray, pos):
        minimal = self.ray_max_distance
        for b in self.borders:
            is_cross, point = b.is_crossing(ray, return_point=True)
            if is_cross:
                if point is not None:
                    distance = np.linalg.norm(pos - point)
                    minimal = min(distance, minimal)
        return minimal
    
    def _get_car_competitor_distances_and_angles(self, car_id):
        competitor_positions = [np.zeros(2) if self.dones[i] else self.positions[i] for i in self.iterate_agents() if i != car_id]
        pos = self.positions[car_id]
        distance_angle_list = [
            (
                np.linalg.norm(pos - other),
                np.arctan2(*(pos - other))
            )
            for other in competitor_positions
        ]
        distance_angle_list.sort(key=lambda d: d[0])
        distance_angle_list = np.array(distance_angle_list)
        return distance_angle_list[:, 0], distance_angle_list[:, 1]
        
    def _get_next_turn(self, car_id):
        for turn in self.turns:
            if turn[0] > self.positions[car_id][0] - 20:
                return turn
        return 0, 0, 0

    def _get_car_obs(self, car_id):
        distances = np.array([
            self._get_ray_collision_distance(r, self.positions[car_id]) for r in self.rays_list[car_id]
        ])
        vision = 1 - distances / self.ray_max_distance
        kernel = np.array([1, 2, 1],)
        vision = np.convolve(vision, kernel / kernel.sum(), 'same')
        turn_angle = self._get_next_turn(car_id)[2]
        turn_info = np.array([np.sin(turn_angle), np.cos(turn_angle)])
        competitor_distances, angles = self._get_car_competitor_distances_and_angles(car_id)
        competitor_distances = 1 - competitor_distances / self.max_recorded_competitor_distance
        competitor_distances = competitor_distances.clip(min=0)
        competitor_angles = np.array([np.sin(angles), np.cos(angles)]).T
        
        return {
            'vision' : vision, 
            'velocity': np.array([self.velocities[car_id] / self.max_velocity]),
            'turn_angle': turn_info,
            'competitor_distances': competitor_distances,
            'competitor_angles': competitor_angles
        }
    
    def _get_obs(self):
        return {i: self._get_car_obs(i) for i in self.iterate_agents() if not self.dones[i]}
    
    def _get_info(self):
        return {}

    def _get_car_cords(self, car_id):
        pos = self.positions[car_id]
        return np.array([
                [pos[0] - self.car_size[0]/2, pos[1] + self.car_size[1]/2], # top left
                [pos[0] + self.car_size[0]/2, pos[1] + self.car_size[1]/2], # top right
                [pos[0] + self.car_size[0]/2, pos[1] - self.car_size[1]/2], # bottom right
                [pos[0] - self.car_size[0]/2, pos[1] - self.car_size[1]/2]  # bottom left
        ])
    
    def _update_car_rays(self, car_id):
        rays = self.rays_list[car_id]
        rays.clear()
        for angle in np.linspace(*self.vision_range, self.rays_count, endpoint=True):
            global_angle = self.directions[car_id] + angle
            ray = Border.from_point_angle(self.positions[car_id], global_angle, self.ray_max_distance)
            rays.append(ray)

    def _get_car_borders(self, car_id):
        cords = self._get_car_cords(car_id)
        borders = []
        for i, (x, y) in enumerate(cords):
            next_vert = cords[(i + 1) % 4]
            borders.append(Border(x, y, *next_vert))
        return borders
    
    def is_collided(self, car_id):
        car_borders = self._get_car_borders(car_id)
        for b in self.borders:
            for c_b in car_borders:
                if b.is_crossing(c_b):
                    return True
        return False
    
    def is_collided_with_other_car(self, car_id):
        car_borders = self._get_car_borders(car_id)
        for other_id in self.iterate_agents():
            if (other_id == car_id) or self.dones[other_id]:
                continue
            for o_b in self._get_car_borders(other_id):
                for b in car_borders:
                    if b.is_crossing(o_b):
                        return True
        return False
            
    def is_finished(self, car_id):
        return self.finish_line.is_crossing(Border(*self.positions[car_id], *self.position_history[car_id][-1]))
    
    def make_action(self, action, car_id):
        angle_change, velocity_change = action['angle_change'], action['velocity_change']
        
        self.directions[car_id] += angle_change[0]
        self.velocities[car_id] = np.clip(self.velocities[car_id] + velocity_change[0], self.min_velocity, self.max_velocity)
        direction = np.deg2rad(self.directions[car_id])
        delta_x = self.velocities[car_id] * np.cos(direction)
        delta_y = self.velocities[car_id] * np.sin(direction)
        
        self.position_history[car_id].append(self.positions[car_id].copy())
        self.positions[car_id] += [delta_x, delta_y]

        self._update_car_rays(car_id)
    
    def _get_car_reward_done_info(self, car_id):
        if self.dones[car_id]:
            return 0, True, {}
        finished = False
        if self.is_collided(car_id) or self.is_collided_with_other_car(car_id):
            reward = -50
            done = True
        elif self.is_finished(car_id):
            reward = (self.max_episode_steps - self.steps_count) / 10
            done = True
            finished = True
        else:
            reward = self.velocities[car_id] / 10
            done = False

        return reward, done, {'finished': finished}
    
    def step(self, actions):
        self.steps_count += 1
        for car_id, action in actions.items():
            self.make_action(action, car_id)
        
        rewards, dones, info = {}, {}, {}
        for car_id in actions:
            rew, dones[car_id], info[car_id] = self._get_car_reward_done_info(car_id)
            self.cumulative_rewards[car_id] += rew
            rewards[car_id] = rew
        self.dones.update(dones)
        dones['__all__'] = (self.steps_count > self.max_episode_steps) or all(self.dones.values())
        
        obs = self._get_obs()
        if self.render_mode is not None:
            self._render_frame(obs)
        
        return obs, rewards, dones, {'__all__': False}, self._get_info()
        
    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    def get_leader(self):
        return max(self.iterate_agents(), key=lambda i: 0 if self.dones[i] else self.positions[i][0])
    
    def _draw_car(self, car_id, canvas, window_center):
        import pygame
        car_borders = self._get_car_borders(car_id)
        
        is_car_done = self.dones[car_id]
        
        for b in car_borders:
            pygame.draw.line(
                canvas,
                (255, 0, 0) if is_car_done else 0,
                *(b.points() - window_center),
                width=3,
            )
        ps = self.position_history[car_id]
        for i in range(len(ps) - 1):
            pygame.draw.line(
                canvas,
                (0, 100, 0),
                ps[i] - window_center,
                ps[i + 1] - window_center,
                width=3,
            )

        
        if not is_car_done and 0:
            for b, intense in zip(self.rays_list[car_id], self._get_car_obs(car_id)['vision']):
                pygame.draw.line(
                    canvas,
                    np.array([255, 100, 50]) * intense,
                    *(b.points() - window_center),
                    width=3,
                )
    
    def _render_frame(self, obs):
        import pygame
        assert self.render_mode is not None
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        leader_car_id = self.get_leader()
        
        window_center = self.positions[leader_car_id] - [self.window_size / 2] * 2

        
        # Draw the track
        for b in self.borders:
            pygame.draw.line(
                canvas,
                (255, 0, 255),
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
        for i, car_id in enumerate(self.iterate_agents()):
            
            self._draw_car(car_id, canvas, window_center)
            
            y_shift = i * 100
            text_surface = self.font.render(f'{self.velocities[car_id]:.1f}', False, (100, 0, 0))
            canvas.blit(text_surface, (100, y_shift))
            text_surface = self.font.render(f'{self.cumulative_rewards[car_id]:.1f}', False, (100, 100, 0))
            canvas.blit(text_surface, (200, y_shift))
        
        
        for x, y, angle in self.turns:
            text_surface = self.font.render(f'{angle / np.pi / 2 * 360:.1f}', False, (100, 100, 100))
            canvas.blit(text_surface, (x - window_center[0], y - window_center[1]))
        
        distances, angles = self._get_car_competitor_distances_and_angles(leader_car_id)
        for i, (d, a) in enumerate(zip(distances, angles)):
            text_surface = self.font.render(f'{d:.1f} {np.rad2deg(a):.1f}', False, (100, 0, 0))
            canvas.blit(text_surface, (300, i * 100))
        
        x, y, angle = self._get_next_turn(leader_car_id)
        text_surface = self.font.render(f'{angle / np.pi / 2 * 360:.1f}', False, (0, 200, 0))
        canvas.blit(text_surface, (x - window_center[0], y - window_center[1]))
        
        
        if self.render_mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            cv2.imwrite(f"frames/{self.frames_count}.jpg", frame)
            self.frames_count += 1


    def reset(self, *, seed=None, options=None):
        # Choose the agent's location uniformly at random
        self.positions = self.generate_start_positions()
        self.directions = {i: 0 for i in self.iterate_agents()} # Theta
        self.velocities = {i: 1 for i in self.iterate_agents()}
        self.rays_list = {i: [] for i in self.iterate_agents()}
        self.dones = {i: False for i in self.iterate_agents()}
        self.cumulative_rewards = {i: 0 for i in self.iterate_agents()}
        self.steps_count = 0
        self.borders, self.finish_line, self.turns = generate_track(turns=self.config['turns_count'])
        # clean the render collection and add the initial frame
        for car_id in self.iterate_agents():
            self._update_car_rays(car_id)
        observation = self._get_obs()
        if self.render_mode is not None:
            self._render_frame(observation)

        info = self._get_info()
        return observation, info
