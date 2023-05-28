import random
import math

from utils import Border


def generate_track(start_x = 0, start_y = 0, track_width = 60, max_step = 400, car_size = 10, turns=10):
    borders = []
    angles = []
    cur_x = start_x + car_size/2
    cur_y = start_y - car_size/2
    prev_road_angle = 0
    y_diff_max = max_step * 0.1
    for i in range(turns):
        if i == 0:
            x_diff = max_step
            y_diff = 0
        else:
            x_diff = random.randint(max_step * 0.8, max_step)
            y_diff = random.randint(-int(y_diff_max), int(y_diff_max))
            y_diff_max += (max_step - y_diff_max) * 0.3
        road_angle = math.atan2(y_diff, x_diff)
        turn_angle = prev_road_angle - road_angle
        prev_road_angle = road_angle
        angles.append([cur_x, cur_y, turn_angle])
        next_x = cur_x + x_diff
        next_y = cur_y + y_diff
        bottom_border = Border(cur_x, cur_y - track_width/2, next_x, next_y - track_width/2)
        top_border = Border(cur_x, cur_y + track_width/2, next_x, next_y + track_width/2)
        cur_x = next_x
        cur_y = next_y
        borders.extend([bottom_border, top_border])
    finish_line = Border(cur_x, cur_y - max_step, cur_x, cur_y + max_step)
    return borders, finish_line, angles
