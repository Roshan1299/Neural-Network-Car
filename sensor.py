import pygame
import math
from utils import *
from settings import *


class Sensor:
    def __init__(self, car):
        self.car = car
        self.ray_count = 5
        self.ray_length = 100
        self.ray_spread = 0.5*math.pi

        self.rays = []
        self.readings = []

    def update(self, road_borders):
        self.ray_cast()
        self.readings = []
        for i in range(len(self.rays)):
            self.readings.append(self.get_reading(self.rays[i], road_borders))

    def get_reading(self, ray, road_borders):
        touches = []
        for i in range(len(road_borders)):
            touch = get_intersection(
                ray[0],
                ray[1],
                road_borders[i][0],
                road_borders[i][1]
            )
            if touch:
                touches.append(touch)
        if len(touches) == 0:
            return None
        else:
            offsets = [e['offset'] for e in touches]
            min_offset = min(offsets)
            return next(e for e in touches if e['offset'] == min_offset)

    def ray_cast(self):
        self.rays = []
        for i in range(self.ray_count):
            ray_angle = lerp(self.ray_spread/2, -
                             self.ray_spread/2, i/(self.ray_count-1))+self.car.angle
            start = Point(self.car.x, self.car.y)
            end = Point(self.car.x - math.sin(ray_angle)*self.ray_length,
                        self.car.y - math.cos(ray_angle)*self.ray_length)

            self.rays.append((start, end))

    def draw(self, screen):
        for i in range(self.ray_count):
            end = self.rays[i][1]
            if self.readings[i] is not None:
                end = self.readings[i]['point']

            if isinstance(end, tuple):
                end = Point(*end)

            pygame.draw.line(
                screen, P_YELLOW, (self.rays[i][0].x, self.rays[i][0].y),
                (end.x, end.y), 2)

            pygame.draw.line(
                screen, BLACK, (self.rays[i][1].x, self.rays[i][1].y),
                (end.x, end.y), 2)
