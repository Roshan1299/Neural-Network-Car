# Dockerization of Neural Network Car Simulation

## Overview

The Neural-Network-Car project has been successfully dockerized to ensure consistent deployment and execution across different environments. This allows users to run the AI car simulation without worrying about dependency conflicts or missing system libraries.

## Docker Components Created

### 1. requirements.txt
Specifies all Python dependencies needed for the simulation:
- pygame: For the game simulation and graphics
- numpy: For numerical operations in neural networks
- matplotlib: For plotting and visualizations
- seaborn: For statistical data visualization

### 2. Dockerfile
- Uses Python 3.10-slim as the base image for a smaller footprint
- Installs all necessary system dependencies for pygame (SDL libraries, etc.)
- Sets up a non-root user for security
- Includes Xvfb virtual display server for headless operation
- Copies application code and installs Python dependencies

### 3. docker-compose.yml
- Configures the container with proper environment variables
- Sets up virtual display environment (DISPLAY=:99) for pygame
- Mounts project directory as volume for data persistence
- Starts Xvfb virtual framebuffer automatically
- Handles headless operation with appropriate display settings

### 4. .dockerignore
Excludes unnecessary files from the Docker build context to optimize build time and image size.

## Benefits of Dockerization

1. **Consistency**: Everyone runs the same environment with the same dependencies
2. **Isolation**: No conflicts with other Python projects or system libraries
3. **Portability**: Runs the same way on different operating systems
4. **Simplified Setup**: One command to get the entire environment running
5. **Reproducibility**: Same results across different machines
6. **Dependency Management**: All required packages are handled automatically

## How to Run

### Prerequisites
- Install Docker: https://docs.docker.com/get-docker/
- Install Docker Compose: Usually included with Docker Desktop

### Running the Simulation
```bash
# Navigate to the project directory
cd /path/to/Neural-Network-Car

# Build and run the container
docker-compose up --build
```

## Special Considerations for GUI Applications

Pygame applications require a display server to function. The Docker setup handles this by:

1. Using Xvfb (X Virtual Framebuffer) to create a virtual display
2. Setting appropriate environment variables (DISPLAY=:99)
3. Using software rendering (SDL_VIDEODRIVER=fbcon) to avoid GPU issues

For systems with direct display access, additional configuration may be needed for X11 forwarding, which is documented in the Test/README.md file.

## Testing the Setup

A validation script has been created in Test/validate_docker_setup.sh that checks:
- All required files exist
- Docker and Docker Compose are installed
- File contents contain necessary components
- Provides instructions for running the application

## Files Created

- `requirements.txt`: Python dependencies
- `Dockerfile`: Container build instructions
- `docker-compose.yml`: Container configuration
- `.dockerignore`: Files to exclude from build context
- `Test/README.md`: Complete documentation
- `Test/validate_docker_setup.sh`: Validation script