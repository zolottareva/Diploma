import random

from utils import Border


def generate_track(start_x = 0, start_y = 0, track_width = 60, max_step = 200, car_size = 10):
    borders = []
    cur_x = start_x + car_size/2
    cur_y = start_y - car_size/2
    for _ in range(10):
        next_x = cur_x + random.randint(max_step * 0.8, max_step)
        next_y = cur_y + random.randint(-60, 40)
        bottom_border = Border(cur_x, cur_y - track_width/2, next_x, next_y - track_width/2)
        top_border = Border(cur_x, cur_y + track_width/2, next_x, next_y + track_width/2)
        cur_x = next_x
        cur_y = next_y
        borders.extend([bottom_border, top_border])
    finish_line = Border(cur_x, cur_y - max_step, cur_x, cur_y + max_step)
    return borders, finish_line