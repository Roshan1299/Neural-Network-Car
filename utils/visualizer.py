import pygame
import numpy as np
from utils.utils import *

class Visualizer:
    def __init__(self):
        self.dash_offset = 0

    def update(self):
        self.dash_offset += 0.05  
        if self.dash_offset >= 1:
            self.dash_offset = 0

    def draw_network(self, screen, network):
        margin = 50
        left = margin
        top = margin
        width = screen.get_width() - margin * 2
        height = screen.get_height() - margin * 2
        level_height = height / len(network.levels)
        for i in range(len(network.levels) - 1, -1, -1):
            level_top = top + lerp(height - level_height, 0, 0.5 if len(
                network.levels) == 1 else i / (len(network.levels) - 1))
            self.draw_level(screen, network.levels[i], left, level_top, width, level_height, [
                            '↑', '←', '→', '↓'] if i == len(network.levels) - 1 else [])

    def draw_level(self, screen, level, left, top, width, height, output_labels):
        right = width + left
        bottom = height + top

        inputs = level.inputs
        outputs = level.outputs
        weights = level.weights
        biases = level.biases

        for i in range(len(inputs)):
            for j in range(len(outputs)):
                start_pos = self.get_node_x(inputs, i, left, right), bottom
                end_pos = self.get_node_x(outputs, j, left, right), top
                color = get_rgba(weights[i][j])
                draw_dashed_line(
                    screen, color[:3], start_pos, end_pos, 2, 10, self.dash_offset)

        node_radius = 32
        shadow_offset = 4

        for i in range(len(inputs)):
            x = self.get_node_x(inputs, i, left, right)
            pygame.draw.circle(
                screen, (0, 33, 48), (x + shadow_offset, bottom + shadow_offset), node_radius)
            pygame.draw.circle(screen, (0, 0, 0), (x, bottom), node_radius)
            pygame.draw.circle(screen, get_rgba(inputs[i])[
                               :3], (x, bottom), int(node_radius * 0.6))

        for i in range(len(outputs)):
            x = self.get_node_x(outputs, i, left, right)
            pygame.draw.circle(
                screen, (0, 33, 48), (x + shadow_offset, top + shadow_offset), node_radius)
            pygame.draw.circle(screen, (0, 0, 0), (x, top), node_radius)
            pygame.draw.circle(screen, get_rgba(outputs[i])[
                               :3], (x, top), int(node_radius * 0.6))
            pygame.draw.circle(screen, get_rgba(biases[i])[
                               :3], (x, top), int(node_radius * 0.8), 2)
            if output_labels and output_labels[i]:
                font = pygame.font.SysFont('Calibri', int(node_radius))
                text_surface = font.render(
                    output_labels[i], True, (206, 17, 38))
                text_rect = text_surface.get_rect(center=(x, top))
                screen.blit(text_surface, text_rect)

    @staticmethod
    def get_node_x(nodes, index, left, right):
        return lerp(left, right, 0.5 if len(nodes) == 1 else index / (len(nodes) - 1))
    
    