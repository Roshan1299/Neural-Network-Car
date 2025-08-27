# NeuroNet_Car

A Python-based simulation of AI-controlled cars navigating a wavy track using **evolutionary neural networks** and **genetic algorithms**. Built with **Pygame**, this project demonstrates how autonomous agents can learn to drive without being explicitly programmed.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/pygame-2.5.0-green?logo=python&logoColor=white)](https://www.pygame.org/news)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/YOUR_USERNAME/NeuroNet_Car/pulls)
[![Website](https://img.shields.io/badge/Live_Analysis-Visit-blue?style=flat&logo=google-chrome)](https://neural-network-car.vercel.app/)

## 🌐 Live Website & Demos

- 🔎 **Analysis Website:** [Visit here](https://neural-network-car.vercel.app/)  
- 🎥 **Initial Demo (AWS S3):** [View Demo](http://neural-net-car.s3-website-us-east-1.amazonaws.com/)  
- 🚀 **Final Demo (Vercel):** [View Demo](https://neural-network-car.vercel.app/)  

## What’s New

- 🎨 Custom Road Drawing → Create your own tracks to test learning in new environments.
- 📊 Plot Trackers → Generate graphs of:
  - Neural decisions
  - Sensor influence
  - Path tracking
  - Weight evolution
  - Mutation impact

- 🌐 Analysis Website → Visualize performance metrics and AI behavior online.
---
## Table of Contents

1. [Features](#features)
2. [Code Structure](#code-structure)
3. [Controls](#controls)
4. [How It Works](#how-it-works)
   - [Neural Network](#neural-network)
   - [Genetic Algorithm](#genetic-algorithm)
   - [Evolution Over Time](#evolution-over-time)
   - [Plot Trackers](#plot-trackers)
5. [Contributing](#contributing)
6. [License](#license)

## Features

- **AI-controlled cars**: Cars are controlled by neural networks that evolve over time using genetic algorithms.
- **Wavy track generation**: The road is dynamically generated with wavy borders, providing a challenging environment for the cars.
- **Sensor system**: Cars are equipped with sensors to detect road borders and avoid collisions.
- **Visualization**: The best-performing car's neural network can be visualized in real-time.
- **Save/Load brains**: Save the best-performing neural network and load it for future simulations.

## Code Structure
The project is organized into multiple files, each handling a specific aspect of the simulation:

```
NEURAL-NETWORK-CAR/
├── assets/                    # Game assets (images, sounds, etc.)
├── controllers/
│   └── controls.py           # Handles user input for car controls
├── environment/
│   ├── road.py               # Generates and manages the wavy track
│   ├── roo.py                # Extra/experimental road generation
│   └── layer_heatmaps/       # Heatmap visualizations
├── models/
│   ├── car.py                # Car movement, collision, and sensors
│   ├── network.py            # Defines neural network structure
│   └── sensor.py             # Simulates car sensors for road detection
├── plot_trackers/
│   ├── decision_tracker.py   # Tracks neural network decision outputs
│   ├── mutation_impact_real.py # Analyzes mutation impact across generations
│   ├── path_tracker.py       # Logs path and fitness progression
│   ├── sensor_analyzer.py    # Analyzes influence of sensors
│   ├── track_editor.py       # Custom track drawing tool
│   └── weight_tracker.py     # Tracks neural network weight evolution
├── utils/
│   ├── utils.py              # Helper functions
│   └── visualizer.py         # Real-time neural network visualization
├── website/                  # Website/analysis dashboard source
├── main.py                   # Main simulation entry point
├── settings.py               # Global constants and configurations
├── custom_track.json         # Saved custom track data
├── sensor_analysis_data.json # Sensor influence logs
├── path_tracking_data.json   # Path tracking logs
├── mutation_data.json        # Mutation evolution logs
├── decision_tracking_data.json # Neural decision logs
├── weight_tracking_data.json # Neural weight logs
├── behavioral_comparison.png # Visualization of behavioral differences
└── .gitignore
```
## Controls
- `R`: Restart simulation with a new generation of cars (mutated from the best saved brain).
- `S`: Save the current best-performing car's neural network (only if it's better than the last saved one).
- `ESC`: Exit the simulation (also auto-saves if a best car exists).

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
### Evolution Over Time
- Over many generations, the cars learn to navigate the track more effectively. The genetic algorithm ensures that the best traits (neural network configurations) are preserved and improved upon.

### Plot Trackers
- Trackers provide detailed graphs of how learning evolves:
- Decision Tracker → Network output over time.
- Path Tracker → Fitness progression & routes.
- Sensor Analyzer → How each sensor affects decisions.
- Weight Tracker → Changes in network weights.
- Mutation Impact → Effect of mutation strength.
- Track Editor → Draw and test your own tracks.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your branch.
4. Submit a pull request.

## License 

This project is licensed under the MIT License. For more details, please see the [LICENSE](LICENSE) file.
