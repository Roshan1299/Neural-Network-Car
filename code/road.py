import pygame
import math
import random
from settings import *


class Road:
    def __init__(self):
        self.center = ((WIDTH) // 2, HEIGHT // 2)
        self.inner_radius = 180
        self.outer_radius = 320
        self.num_segments = 50
        self.stretch_x = 1.5
        self.wavy_amplitude = 120
        self.wavy_frequency = 2

        self.inner_points, self.outer_points = self.create_wavy_track()
        self.borders = self.create_borders()

    def create_wavy_track(self):
        inner_points = []
        outer_points = []
        angle_step = 2 * math.pi / self.num_segments
        for i in range(self.num_segments):
            theta = i * angle_step

            # Add a wavy variation
            wave_variation = self.wavy_amplitude * \
                math.sin(self.wavy_frequency * theta)

            inner_radius_variation = self.inner_radius + \
                random.uniform(-10, 10) + wave_variation
            outer_radius_variation = self.outer_radius + \
                random.uniform(-10, 10) + wave_variation

            x_inner = self.center[0] + inner_radius_variation * \
                math.cos(theta) * self.stretch_x
            y_inner = self.center[1] + inner_radius_variation * math.sin(theta)
            inner_points.append((x_inner, y_inner))

            x_outer = self.center[0] + outer_radius_variation * \
                math.cos(theta) * self.stretch_x
            y_outer = self.center[1] + outer_radius_variation * math.sin(theta)
            outer_points.append((x_outer, y_outer))

        return inner_points, outer_points

    def create_borders(self):
        borders = []
        for i in range(self.num_segments):
            next_i = (i + 1) % self.num_segments
            inner_start = self.inner_points[i]
            inner_end = self.inner_points[next_i]
            outer_start = self.outer_points[i]
            outer_end = self.outer_points[next_i]

            borders.append((inner_start, inner_end))
            borders.append((outer_start, outer_end))

        return borders

    def draw(self, screen):
        pygame.draw.polygon(screen, (60,60,60), self.outer_points)
        pygame.draw.polygon(screen, BLACK, self.inner_points, 0)

        pygame.draw.polygon(screen, (34, 139, 34), self.inner_points, 4)
        pygame.draw.polygon(screen, (34, 139, 34), self.outer_points, 4)
