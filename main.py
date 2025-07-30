import pygame
import json
import os
import random
import numpy as np  # Add this import
from settings import *
from models.car import Car
from environment.road import Road
from models.network import NeuralNetwork
from utils.visualizer import Visualizer
from weight_tracker import WeightTracker
from decision_tracker import DecisionTracker
from sensor_analyzer import SensorAnalyzer
from path_tracker import PathTracker  # NEW: Add path tracker import

generation = 1
num_cars = 10  # Total cars per generation
current_mutation_rate = 0.1  # Default mutation rate
mutation_data = []  # Global mutation tracking data

weight_tracker = WeightTracker()
decision_tracker = DecisionTracker()
sensor_analyzer = SensorAnalyzer()
path_tracker = PathTracker()  # NEW: Initialize path tracker

def load_mutation_data():
    """Load existing mutation tracking data"""
    if os.path.exists('mutation_data.json'):
        with open('mutation_data.json', 'r') as f:
            return json.load(f)
    return []

def save_mutation_data():
    """Save mutation tracking data"""
    with open('mutation_data.json', 'w') as f:
        json.dump(mutation_data, f, indent=2)

def save_state(car, generation):
    brain_state = car.brain.get_state()
    save_data = {
        "generation": generation,
        "brain": brain_state
    }
    with open("save_state.json", "w") as f:
        json.dump(save_data, f)
    
    # Track weights and save data
    weight_tracker.track_weights(car.brain, generation)
    weight_tracker.save_data()
    decision_tracker.save_data()
    sensor_analyzer.save_data()
    path_tracker.save_data()  # NEW: Save path data
    print(f"Saved state, tracked weights, decisions, sensors, and paths for generation {generation}")

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
    global current_mutation_rate
    best_brain, gen = load_state()
    if best_brain:
        for i, car in enumerate(cars):
            car.brain = best_brain.clone()
            if i != 0:
                NeuralNetwork.mutate(car.brain, current_mutation_rate)
    return gen

def generate_cars(num_cars):
    cars = pygame.sprite.Group([
        Car(1050, 550, 40, 80, "AI")
        for _ in range(num_cars)
    ])
    return cars

def reset_game():
    global cars, best_car, road, generation, mutation_data, current_mutation_rate, path_tracker
    
    # Store final generation stats BEFORE resetting
    if 'best_car' in globals() and best_car:
        path_tracker.finalize_generation(
            generation - 1,  # Current generation that's ending
            best_car.distance_traveled,
            crashed=best_car.damaged if hasattr(best_car, 'damaged') else False
        )
    
    # Calculate performance metrics BEFORE resetting
    if 'cars' in globals() and cars:
        # Get fitness values from all cars
        fitness_values = [car.distance_traveled for car in cars]
        current_best_fitness = max(fitness_values) if fitness_values else 0
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
        cars_survived = len([car for car in cars if car.moving and car.distance_traveled > 0])
        
        # Calculate improvement from previous generation
        previous_best = mutation_data[-1]['best_fitness'] if mutation_data else 0
        performance_improvement = current_best_fitness - previous_best
        
        # Calculate population diversity (std dev of fitness)
        fitness_std = np.std(fitness_values) if len(fitness_values) > 1 else 0
        
        # Store generation data
        generation_data = {
            'generation': generation,
            'mutation_rate': current_mutation_rate,
            'best_fitness': current_best_fitness,
            'avg_fitness': avg_fitness,
            'performance_improvement': performance_improvement,
            'population_diversity': fitness_std,
            'cars_survived': cars_survived,
            'cars_beat_previous': len([car for car in cars if car.distance_traveled > previous_best]),
            'fitness_std': fitness_std
        }
        mutation_data.append(generation_data)
        save_mutation_data()
        print(f"Tracked generation {generation}: Best={current_best_fitness:.1f}, Improvement={performance_improvement:.1f}, MutRate={current_mutation_rate:.3f}")
    
    # Reset for next generation
    road = Road()
    path_tracker.set_track_boundaries(road)  # NEW: Store track for visualization
    cars = generate_cars(num_cars)
    generation += 1
    apply_brain_to_cars(cars)

def main():
    global cars, best_car, road, generation, num_cars, weight_tracker, decision_tracker, sensor_analyzer, path_tracker, mutation_data, current_mutation_rate
    pygame.init()
    screen = pygame.display.set_mode((1200, 900))
    pygame.display.set_caption('NeuroNet')
    icon = pygame.image.load('assets/car_1.png')
    pygame.display.set_icon(icon)

    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    net_width = 600

    road = Road()
    path_tracker.set_track_boundaries(road)  # NEW: Set initial track boundaries
    cars = generate_cars(num_cars)
    generation = apply_brain_to_cars(cars)
    
    # Load previous tracking data
    weight_tracker.load_data()
    decision_tracker.load_data()
    sensor_analyzer.load_data()
    path_tracker.load_data()  # NEW: Load path data
    
    # Load mutation tracking data
    mutation_data = load_mutation_data()
    print(f"Loaded {len(mutation_data)} generations of mutation data")
    print("All trackers initialized. Press 'H' for weights, 'D' for decisions, 'Z' for sensors, 'M' for mutation, 'P' for paths!")

    net_screen = pygame.Surface((net_width, HEIGHT))
    net_screen.fill(BLACK)

    road_screen = pygame.Surface((WIDTH, HEIGHT))
    road_screen.fill(BLACK)

    visualizer = Visualizer()

    running = True
    frame_count = 0
    
    while running:
        frame_count += 1
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
                elif event.key == pygame.K_h:  # Weight heatmaps
                    print("Generating weight evolution heatmaps...")
                    try:
                        weight_tracker.create_heatmap()
                        weight_tracker.create_layer_specific_heatmaps()
                        weight_tracker.create_weight_change_heatmap()
                        
                        stats = weight_tracker.get_statistics()
                        print("\n=== Weight Evolution Statistics ===")
                        print(f"Total connections tracked: {stats.get('total_connections', 0)}")
                        print(f"Generations tracked: {stats.get('generations_tracked', 0)}")
                        if 'weight_range' in stats:
                            wr = stats['weight_range']
                            print(f"Weight range: {wr['min']:.3f} to {wr['max']:.3f}")
                            print(f"Average weight: {wr['mean']:.3f} (±{wr['std']:.3f})")
                        print("Weight heatmaps saved successfully!")
                    except Exception as e:
                        print(f"Error generating weight heatmaps: {e}")
                
                elif event.key == pygame.K_d:  # Decision pattern analysis
                    print("Generating decision pattern analysis...")
                    try:
                        decision_tracker.create_radar_chart()
                        decision_tracker.create_evolution_timeline()
                        decision_tracker.create_behavioral_comparison()
                        
                        # Print insights
                        insights = decision_tracker.get_insights()
                        print("\n=== Behavioral Evolution Insights ===")
                        print(insights)
                        print("Decision pattern charts saved successfully!")
                    except Exception as e:
                        print(f"Error generating decision patterns: {e}")
                
                elif event.key == pygame.K_z:  # Sensor analysis
                    print("Generating sensor utilization analysis...")
                    try:
                        sensor_analyzer.create_sensor_importance_chart()
                        sensor_analyzer.create_sensor_redundancy_analysis()
                        sensor_analyzer.create_real_time_sensor_display()
                        
                        # Print insights
                        insights = sensor_analyzer.get_sensor_insights()
                        print("\n=== Sensor Utilization Insights ===")
                        print(insights)
                        print("Sensor analysis charts saved successfully!")
                    except Exception as e:
                        print(f"Error generating sensor analysis: {e}")
                
                elif event.key == pygame.K_m:  # Mutation impact analysis
                    print("Generating mutation impact analysis...")
                    try:
                        if len(mutation_data) >= 5:  # Need at least 5 generations
                            # Run the mutation analysis (you'll need to create this file)
                            import subprocess
                            result = subprocess.run(['python3', 'mutation_impact_real.py'], 
                                                  capture_output=True, text=True)
                            if result.returncode == 0:
                                print("Mutation impact analysis completed successfully!")
                                print("Check the generated charts and recommendations.")
                            else:
                                print(f"Error in mutation analysis: {result.stderr}")
                        else:
                            print(f"Need at least 5 generations for analysis. Current: {len(mutation_data)}")
                            print("Run more generations (press 'R') to collect data.")
                    except Exception as e:
                        print(f"Error generating mutation analysis: {e}")
                
                elif event.key == pygame.K_p:  # NEW: Path visualization
                    print("Generating path tracking visualizations...")
                    try:
                        path_tracker.save_data()  # Save current data
                        
                        print("Creating ghost racing visualization...")
                        path_tracker.create_ghost_racing_visualization()
                        
                        print("Creating heat trail visualization...")
                        path_tracker.create_heat_trail_visualization()
                        
                        print("Creating performance overlay...")
                        path_tracker.create_performance_overlay_visualization()
                        
                        print("Creating evolution comparison...")
                        path_tracker.create_evolution_comparison()
                        
                        # Print summary statistics
                        stats = path_tracker.create_summary_statistics()
                        
                        print("\nAll path visualizations saved successfully!")
                        print("Files created:")
                        print("- path_evolution_ghost_racing.png")
                        print("- path_heat_trail.png") 
                        print("- path_performance_overlay.png")
                        print("- path_evolution_comparison.png")
                        
                    except Exception as e:
                        print(f"Error generating path visualizations: {e}")
                
                # Mutation rate adjustment keys
                elif event.key == pygame.K_1:  # Low mutation
                    current_mutation_rate = 0.05
                    print(f"Mutation rate set to: {current_mutation_rate}")
                elif event.key == pygame.K_2:  # Medium mutation  
                    current_mutation_rate = 0.1
                    print(f"Mutation rate set to: {current_mutation_rate}")
                elif event.key == pygame.K_3:  # High mutation
                    current_mutation_rate = 0.2
                    print(f"Mutation rate set to: {current_mutation_rate}")

            for car in cars:
                car.controls.handle_event(event)

        road_screen.fill(BLACK)
        road.draw(road_screen)
        screen.blit(road_screen, (0, 0))

        for car in cars:
            car.update(screen, road.borders)

        best_car = max((car for car in cars if car.moving),
                       key=lambda car: car.distance_traveled, default=None)

        # Sample path data from the best car
        if best_car and best_car.moving:
            path_tracker.sample_path(best_car, generation, frame_count)
            
            try:
                # Get sensor readings for decision tracking
                sensor_readings = best_car.get_sensor_distances() if hasattr(best_car, 'get_sensor_distances') else [0]*5
                decision_tracker.sample_decision(best_car, generation, sensor_readings)
                
                # Sample sensor data for sensor analysis
                sensor_analyzer.sample_sensor_data(best_car, generation, frame_count)
            except Exception as e:
                # Fallback if sensor access fails
                pass

        if best_car:
            best_car.draw(screen, True)

        net_screen.fill(BLACK)
        visualizer.update()
        screen.blit(net_screen, (1200, 0))

        gen_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
        screen.blit(gen_text, (20, 20))
        
        # Display current mutation rate
        mut_text = font.render(f"Mutation Rate: {current_mutation_rate:.3f}", True, (255, 200, 100))
        screen.blit(mut_text, (20, 60))
        
        # Display tracking info
        if len(weight_tracker.weight_history) > 0:
            weight_text = font.render(f"Weights tracked: {len(weight_tracker.weight_history)} gens", True, (200, 200, 200))
            screen.blit(weight_text, (20, 100))
        
        if len(decision_tracker.decision_history) > 0:
            decision_text = font.render(f"Decisions tracked: {len(decision_tracker.decision_history)} gens", True, (200, 200, 200))
            screen.blit(decision_text, (20, 140))
        
        # Display sensor tracking info
        if len(sensor_analyzer.sensor_history) > 0:
            total_samples = sum(len(data) for data in sensor_analyzer.sensor_history.values())
            sensor_text = font.render(f"Sensor samples: {total_samples}", True, (200, 200, 200))
            screen.blit(sensor_text, (20, 180))
        
        # Display mutation data tracking
        if len(mutation_data) > 0:
            mutation_text = font.render(f"Mutation data: {len(mutation_data)} gens", True, (200, 200, 200))
            screen.blit(mutation_text, (20, 220))
        
        # NEW: Display path tracking info
        if len(path_tracker.path_history) > 0:
            path_text = font.render(f"Paths tracked: {len(path_tracker.path_history)} gens", True, (200, 200, 200))
            screen.blit(path_text, (20, 260))
            
        # Updated instructions text
        instructions_text = pygame.font.SysFont(None, 20).render(
            "H=Heatmaps, D=Decisions, Z=Sensors, M=Mutation, P=Paths, 1/2/3=MutRate", 
            True, (150, 150, 150)
        )
        screen.blit(instructions_text, (20, 300))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()