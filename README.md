# NeuroNet_Car

This project is a Python-based simulation of AI-controlled cars navigating a wavy track. The cars are controlled by neural networks, and the project demonstrates the use of genetic algorithms to evolve the AI's driving behavior. Built using **Pygame**, this project is a fun and educational way to explore AI, neural networks, and simulation development.

## Features

- **AI-controlled cars**: Cars are controlled by neural networks that evolve over time using genetic algorithms.
- **Wavy track generation**: The road is dynamically generated with wavy borders, providing a challenging environment for the cars.
- **Sensor system**: Cars are equipped with sensors to detect road borders and avoid collisions.
- **Visualization**: The best-performing car's neural network can be visualized in real-time.
- **Save/Load brains**: Save the best-performing neural network and load it for future simulations.

## Code Structure
The project is organized into multiple files, each handling a specific aspect of the simulation:

- `car.py`: Implements the Car class, which handles the car's movement, collision detection, and AI behavior. It also manages the car's sensors and neural network.
- `controls.py`: Defines the Controls class, which handles user input (keyboard controls) and maps them to car movements (forward, left, right, reverse).
- `main.py`: Contains the main game loop, initializes the simulation, and manages the rendering of cars and the road. It also handles saving and loading the best-performing neural network.
- `network.py`: Implements the NeuralNetwork and Level classes, which define the structure and behavior of the neural network used by the AI cars. It includes functions for feedforward propagation, mutation, and cloning.
- `road.py`: Defines the Road class, which generates the wavy track and manages the road's borders. It also provides functions for drawing the road on the screen.
- `sensor.py`: Implements the Sensor class, which simulates the car's sensors to detect distances to the road borders. This data is used as input for the neural network.
- `settings.py`: Contains constants and configuration settings for the simulation, such as screen dimensions, colors, and neural network parameters.
- `visualizer.py`: Provides visualization tools for the neural network, allowing you to see how the AI is making decisions in real-time.
- `best_brain.json`: A JSON file that stores the state of the best-performing neural network. This file is generated when the user saves the brain and can be loaded in future simulations.

## How it works

### Neural Network

The AI cars are controlled by a **feedforward neural network**, which is a type of artificial neural network where connections between nodes do not form cycles. Here's how it works:

#### Input Layer
- The neural network takes inputs from the car's sensors. Each sensor measures the distance to the nearest road border in a specific direction.
- These distances are **normalized** (scaled to a range of 0 to 1) to ensure consistent input values for the neural network.

#### Hidden Layers
- The network has one or more hidden layers that process the inputs. Each node in the hidden layers applies a weighted sum of its inputs, followed by an **activation function** (in this case, a simple threshold function is used).
- The hidden layers enable the network to learn complex patterns and make decisions based on the sensor data.

#### Output Layer
- The output layer produces **four values**, each corresponding to a control signal:
  - **Forward**: Accelerate the car.
  - **Left**: Steer the car to the left.
  - **Right**: Steer the car to the right.
  - **Reverse**: Slow down or reverse the car.
- The output values determine the car's movement and steering.

### Genetic Algorithm

The genetic algorithm is used to evolve the neural networks controlling the cars. It mimics the process of natural selection to improve the performance of the AI over generations. Here's how it works:

#### Initial Population
- A population of cars is created, each with a **randomly initialized neural network**.

#### Fitness Evaluation
- The performance of each car is evaluated based on how far it travels without crashing into the road borders. This distance is used as the **fitness score**.
- The car with the highest fitness score is considered the **best-performing car**.

#### Selection
- The best-performing car is selected to be the **"parent"** for the next generation. Its neural network is cloned and used as the base for the new population.

#### Mutation
- To introduce variability and explore new solutions, the cloned neural network is **mutated**. Mutation involves randomly adjusting the weights and biases of the network by a small amount.
- The **mutation rate** determines how much the network is altered. A higher mutation rate leads to more exploration but may disrupt good solutions, while a lower rate focuses on refining existing solutions.

#### Next Generation
- A new population of cars is created using the mutated neural network. This process is repeated for multiple generations, allowing the cars to gradually improve their driving behavior.

#### Evolution Over Time
- Over many generations, the cars learn to navigate the track more effectively. The genetic algorithm ensures that the best traits (neural network configurations) are preserved and improved upon.

## License 

This project is licensed under the MIT License. For more details, please see the [LICENSE](LICENSE) file.
