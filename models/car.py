import pygame
import math
from settings import *
from controllers.controls import Controls
from models.sensor import Sensor, Point
from utils.utils import *
from models.network import NeuralNetwork

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, control_type, max_speed=4):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.image_normal = pygame.image.load(
            "assets/car_1.png").convert_alpha()
        self.image_normal = pygame.transform.smoothscale(
            self.image_normal, (self.width, self.height))
        self.image_damaged = pygame.image.load(
            "assets/car_c.png").convert_alpha()
        self.image_damaged = pygame.transform.smoothscale(
            self.image_damaged, (self.width, self.height))

        self.image = self.image_normal
        self.rect = self.image.get_rect()

        self.angle = 0
        self.speed = 0
        self.acceleration = 0.2
        self.max_speed = max_speed
        self.friction = 0.05
        self.angular_velocity = 0.03
        self.damaged = False
        self.moving = False
        self.distance_traveled = 0

        self.use_brain = control_type == "AI"

        self.controls = Controls(control_type)
        self.sensor = Sensor(self) if self.use_brain else None
        self.brain = NeuralNetwork([self.sensor.ray_count, 6, 4])

    def update(self, screen, road_borders):
        self.polygon = self.create_polygon()
        self.damaged = self.assess_damage(road_borders)
        if self.damaged:
            self.image = self.image_damaged
        if not self.damaged:
            self.move()
            self.moving = True
            self.rotate_center(screen)
        else:
            self.rotate_center(screen)

        if self.sensor:
            self.sensor.update(road_borders)
            inputs = [
                reading["offset"] if reading is not None else 0 for reading in self.sensor.readings]
            normalized_inputs = self.brain.preprocess_inputs(inputs)
            outputs = NeuralNetwork.feed_forward(normalized_inputs, self.brain)
            self.controls.forward = outputs[0]
            self.controls.left = outputs[1]
            self.controls.right = outputs[2]
            self.controls.reverse = outputs[3]
        # self.draw(screen)

    def rotate_center(self, screen):
        rotated_image = pygame.transform.rotate(
            self.image, math.degrees(self.angle))
        self.rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, self.rect)

    def assess_damage(self, road_borders):
        polygon_points = [Point(x, y) for x, y in self.polygon]
        for border in road_borders:
            border_points = [(p[0], p[1]) for p in border]
            if polys_intersect(polygon_points, border_points):
                return True
        return False

    def move(self):
        if self.controls.forward:
            self.speed += self.acceleration
        if self.controls.reverse:
            self.speed -= self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed:
            self.speed = -self.max_speed
        if self.speed > 0:
            self.speed -= self.friction
        if self.speed < 0:
            self.speed += self.friction
        if abs(self.speed) < self.friction:
            self.speed = 0

        if self.speed != 0:
            flip = 1 if self.speed > 0 else -1
            if self.controls.left:
                self.angle += self.angular_velocity * flip
            if self.controls.right:
                self.angle -= self.angular_velocity * flip

        self.x -= math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed

        self.distance_traveled += self.speed

    def create_polygon(self):
        points = []
        rad = math.hypot(self.rect[2]-35, self.rect[3]-15)/2
        alpha = math.atan2(self.rect[2]-35, self.rect[3]-15)

        points.append((self.x - math.sin(self.angle - alpha) * rad,
                      self.y - math.cos(self.angle - alpha) * rad))
        points.append((self.x - math.sin(self.angle + alpha) * rad,
                      self.y - math.cos(self.angle + alpha) * rad))
        points.append((self.x - math.sin(math.pi + self.angle - alpha)
                      * rad, self.y - math.cos(math.pi + self.angle - alpha) * rad))
        points.append((self.x - math.sin(math.pi + self.angle + alpha)
                      * rad, self.y - math.cos(math.pi + self.angle + alpha) * rad))

        return points

    def draw(self, screen, draw_sensor=False):
        pygame.draw.polygon(
            screen, P_RED, [(p[0], p[1]) for p in self.polygon], 1)
        # if self.sensor and draw_sensor:
        #     self.sensor.draw(screen)
