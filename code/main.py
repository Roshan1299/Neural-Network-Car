import pygame
import json
import os
import random
from settings import *
from car import Car
from road import Road
from network import NeuralNetwork
from visualizer import Visualizer

generation = 1
num_cars = 10  # Total cars per generation

def save_state(car, generation):
    brain_state = car.brain.get_state()
    save_data = {
        "generation": generation,
        "brain": brain_state
    }
    with open("save_state.json", "w") as f:
        json.dump(save_data, f)

def load_state():
    if os.path.exists('save_state.json'):
        with open('save_state.json', 'r') as f:
            data = json.load(f)
            brain_state = data["brain"]
            gen = data.get("generation", 1)

            neural_network = NeuralNetwork([len(brain_state["levels"][0]["inputs"])] +
                                           [len(level["outputs"]) for level in brain_state["levels"]])
            neural_network.set_state(brain_state)
            return neural_network, gen
    return None, 1

def apply_brain_to_cars(cars):
    best_brain, gen = load_state()
    if best_brain:
        for i, car in enumerate(cars):
            car.brain = best_brain.clone()
            if i != 0:
                NeuralNetwork.mutate(car.brain, 0.1)
    return gen

def generate_cars(num_cars):
    cars = pygame.sprite.Group([
        Car(random.uniform(1100, 1150), random.uniform(470, 500), 40, 80, "AI")
        for _ in range(num_cars)
    ])
    return cars

def reset_game():
    global cars, best_car, road, generation
    road = Road()
    cars = generate_cars(num_cars)
    generation += 1
    apply_brain_to_cars(cars)

def main():
    global cars, best_car, road, generation, num_cars
    pygame.init()
    screen = pygame.display.set_mode((1400, 700))
    pygame.display.set_caption('NeuroNet')
    icon = pygame.image.load('assets/car_1.png')
    pygame.display.set_icon(icon)

    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    net_width = 600

    road = Road()
    cars = generate_cars(num_cars)
    generation = apply_brain_to_cars(cars)

    net_screen = pygame.Surface((net_width, HEIGHT))
    net_screen.fill(BLACK)

    road_screen = pygame.Surface((WIDTH, HEIGHT))
    road_screen.fill(BLACK)

    visualizer = Visualizer()

    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if best_car:
                    save_state(best_car, generation)
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if best_car:
                        save_state(best_car, generation)
                    running = False
                elif event.key == pygame.K_r:
                    if best_car:
                        save_state(best_car, generation)
                    reset_game()
                elif event.key == pygame.K_s:
                    if best_car:
                        save_state(best_car, generation)

            for car in cars:
                car.controls.handle_event(event)

        road_screen.fill(BLACK)
        road.draw(road_screen)
        screen.blit(road_screen, (0, 0))

        for car in cars:
            car.update(screen, road.borders)

        best_car = max((car for car in cars if car.moving),
                       key=lambda car: car.distance_traveled, default=None)

        if best_car:
            best_car.draw(screen, True)

        net_screen.fill(BLACK)
        visualizer.update()
        screen.blit(net_screen, (1200, 0))

        gen_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
        screen.blit(gen_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
