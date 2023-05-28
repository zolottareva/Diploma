import numpy as np

class Border:
    def __init__(self, x1, y1, x2, y2):
        self.line = np.array([x1, y1, x2, y2])
    
    @classmethod
    def from_point_angle(cls, point, angle, distance):
        angle = np.deg2rad(angle)
        p2 = np.array(point) + np.array([np.cos(angle), np.sin(angle)]) * distance
        return cls(*point, *p2)

    def get_crossing_point(self, other):
        k_self, b_self = self.count_k_b()
        k_border, b_border = other.count_k_b()
        if k_self == k_border:
            return None # parallel
        if k_self == float('inf'):
            x_crossing = self.line[0]
            y_crossing = k_border * x_crossing + b_border
        elif k_border == float('inf'):
            x_crossing = other.line[0]
            y_crossing = k_self * x_crossing + b_self
        else:
            x_crossing = (b_self - b_border) / (k_border - k_self)
            y_crossing = k_self * x_crossing + b_self
        return x_crossing, y_crossing

    def is_crossing(self, border, return_point=False):
        point = self.get_crossing_point(border)
        if point is None:
            is_crossing = False
        else:
            is_crossing = self._check_if_inside(*point) and border._check_if_inside(*point)
        if return_point:
            return is_crossing, point
        return is_crossing

    def points(self):
        return np.array([self.line[:2], self.line[2:]])

    def count_k_b(self):
        if self.line[2] == self.line[0]: # vertical line
            return float('inf'), self.line[0]
        k = (self.line[3] - self.line[1])/(self.line[2]-self.line[0]) # (y2-y1)/(x2-x1)
        b = self.line[1] - k * self.line[0] # y1 - k * x1
        return k, b

    def _check_if_inside(self, x, y):
        if self.line[0] != self.line[2]:
            return x >= min(self.line[0], self.line[2]) and x <= max(self.line[0], self.line[2])
        return y >= min(self.line[1], self.line[3]) and y <= max(self.line[1], self.line[3])
    