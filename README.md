# NeuroNet_Car

This project is a Python-based simulation of AI-controlled cars navigating a wavy track. The cars are controlled by neural networks, and the project demonstrates the use of genetic algorithms to evolve the AI's driving behavior. Built using **Pygame**, this project is a fun and educational way to explore AI, neural networks, and simulation development.

## Features

- **AI-controlled cars**: Cars are controlled by neural networks that evolve over time using genetic algorithms.
- **Wavy track generation**: The road is dynamically generated with wavy borders, providing a challenging environment for the cars.
- **Sensor system**: Cars are equipped with sensors to detect road borders and avoid collisions.
- **Visualization**: The best-performing car's neural network can be visualized in real-time.
- **Save/Load brains**: Save the best-performing neural network and load it for future simulations.

## How it works

### Neural Network
The AI cars are controlled by a simple feedforward neural network. The network takes sensor inputs (distance to road borders) and outputs control signals (forward, left, right, reverse).
The neural network is trained using a genetic algorithm to improve its performance over time.

### Genetic Algorithm
The simulation uses a genetic algorithm to evolve the AI cars:
- Selection: The best-performing car (the one that travels the farthest without crashing) is selected.
- Mutation: The neural network of the best car is cloned and slightly mutated to introduce variability.
- Next Generation: A new set of cars is created using the mutated neural network, ensuring that each generation improves over the previous one.
### Road Generation
The road is generated as a wavy track with inner and outer borders. The cars must navigate the track without colliding with the borders.
