import pygame
import json
import os
import random
from settings import *
from car import Car
from road import Road
from network import NeuralNetwork
from visualizer import Visualizer


def save_brain(car):
    brain_state = car.brain.get_state()
    if os.path.exists('best_brain.json'):
        with open('best_brain.json', 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(brain_state)

    with open('best_brain.json', 'w') as f:
        json.dump(data, f)


def load_brain():
    if os.path.exists('best_brain.json'):
        with open('best_brain.json', 'r') as f:
            data = json.load(f)
            brain_state = data[-1]
            neural_network = NeuralNetwork([len(brain_state["levels"][0]["inputs"])] + [len(
                level["outputs"]) for level in brain_state["levels"]])
            neural_network.set_state(brain_state)
            return neural_network
    return None


def apply_brain_to_cars(cars):
    best_brain = load_brain()
    if best_brain:
        for i, car in enumerate(cars):
            car.brain = best_brain.clone()
            if i != 0:
                NeuralNetwork.mutate(car.brain, 0.1)


def generate_cars(num_cars): 
    cars = pygame.sprite.Group([
        Car(random.uniform(1100, 1150),random.uniform(470, 500), 40, 80, "AI")
        for i in range(num_cars)
    ])    
    
    return cars


def reset_game():
    global cars, best_car, road
    road = Road()
    cars = generate_cars(num_cars)
    apply_brain_to_cars(cars)
    
def main():
    global cars, best_car, road, num_cars
    pygame.init()
    screen = pygame.display.set_mode((1400, 700))
    pygame.display.set_caption('NeuroNet')
    icon = pygame.image.load('img/car_1.png')
    pygame.display.set_icon(icon)

    clock = pygame.time.Clock()
    net_width = 600

    road = Road()
    cars = generate_cars(num_cars)

    net_screen = pygame.Surface((net_width, HEIGHT))
    net_screen.fill(BLACK)

    road_screen = pygame.Surface((WIDTH, HEIGHT))
    road_screen.fill(BLACK)

    visualizer = Visualizer()

    while True:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_game()
                apply_brain_to_cars(cars)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                save_brain(best_car)
            for car in cars:
                car.controls.handle_event(event)

        road_screen.fill(BLACK)
        road.draw(road_screen)
        screen.blit(road_screen, (0, 0))

        for car in cars:
            car.update(screen, road.borders)
            # car.draw(screen)

        best_car = max((car for car in cars if car.moving),
                       key=lambda car: car.distance_traveled, default=None)

        best_car.draw(screen, True)

        net_screen.fill(BLACK)
        visualizer.update()
        #visualizer.draw_network(net_screen, best_car.brain)
        screen.blit(net_screen, (1200, 0))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
