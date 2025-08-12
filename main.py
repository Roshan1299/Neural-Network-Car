import pygame
import json
import os
import random
import math
import numpy as np
from settings import *
from models.car import Car
from environment.road import Road 
from models.network import NeuralNetwork
from utils.visualizer import Visualizer
from weight_tracker import WeightTracker
from decision_tracker import DecisionTracker
from sensor_analyzer import SensorAnalyzer
from path_tracker import PathTracker

generation = 1
num_cars = 10  # Total cars per generation
current_mutation_rate = 0.1  # Default mutation rate
mutation_data = []  # Global mutation tracking data

weight_tracker = WeightTracker()
decision_tracker = DecisionTracker()
sensor_analyzer = SensorAnalyzer()
path_tracker = PathTracker()

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
    path_tracker.save_data()
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

def get_starting_position(road):
    """Get a good starting position on the track"""
    if not road.drawing_complete or len(road.centerline_points) < 4:
        # Default position if no custom track
        return 600, 450, 0  # x, y, angle
    
    # Use first few points of centerline to determine starting position and angle
    start_point = road.centerline_points[0]
    next_point = road.centerline_points[1] if len(road.centerline_points) > 1 else road.centerline_points[0]
    
    # Calculate angle from start to next point
    dx = next_point[0] - start_point[0]
    dy = next_point[1] - start_point[1]
    angle = math.atan2(dx, dy)
    
    return start_point[0], start_point[1], angle

def generate_cars(num_cars, road):
    """Generate cars at the starting position"""
    start_x, start_y, start_angle = get_starting_position(road)
    
    cars_list = []
    for i in range(num_cars):
        # Spread cars slightly to avoid overlapping
        offset_x = random.uniform(-20, 20)
        offset_y = random.uniform(-20, 20)
        car = Car(start_x + offset_x, start_y + offset_y, 40, 80, "AI")
        car.angle = start_angle  # Set initial angle to face track direction
        cars_list.append(car)
    
    return pygame.sprite.Group(cars_list)

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
    
    # Reset for next generation - keep the same road
    cars = generate_cars(num_cars, road)
    generation += 1
    apply_brain_to_cars(cars)

def enter_track_editor_mode(screen, road, font):
    """Enter track editing mode within the game"""
    print("Entering track editor mode...")
    print("Draw your track, then press ESC to return to racing!")
    
    editing = True
    clock = pygame.time.Clock()
    
    while editing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to quit entire game
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    editing = False
                elif event.key == pygame.K_e:
                    editing = False
            
            # Handle road editing
            road.handle_event(event)
        
        # Draw everything
        screen.fill(GREEN)
        road.draw(screen)
        road.draw_instructions(screen, font)
        
        # Add exit instruction
        exit_text = font.render("Press E or ESC to exit editor and start racing!", True, WHITE)
        exit_rect = exit_text.get_rect(center=(WIDTH//2, 50))
        
        # Semi-transparent background for text
        overlay = pygame.Surface((exit_rect.width + 20, exit_rect.height + 10))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        overlay_rect = overlay.get_rect(center=(WIDTH//2, 50))
        screen.blit(overlay, overlay_rect)
        screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    return True  # Continue with game

def main():
    global cars, best_car, road, generation, num_cars, weight_tracker, decision_tracker, sensor_analyzer, path_tracker, mutation_data, current_mutation_rate
    
    pygame.init()
    screen = pygame.display.set_mode((1200, 900))
    pygame.display.set_caption('NeuroNet - AI Racing with Custom Tracks')
    
    try:
        icon = pygame.image.load('assets/car_1.png')
        pygame.display.set_icon(icon)
    except:
        print("Could not load icon, continuing without it...")

    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    net_width = 600

    # Initialize road - try to load custom track first
    road = Road(load_saved=True)
    
    # Check if we have a valid track
    if not road.drawing_complete:
        print("=" * 60)
        print("WELCOME TO AI RACING WITH CUSTOM TRACKS!")
        print("=" * 60)
        print()
        print("No custom track found! You have options:")
        print("1. Press 'E' in-game to create a custom track")
        print("2. Continue with default track")
        print("3. Run 'python track_editor.py' separately")
        print()
        
        # Show welcome screen
        welcome_active = True
        welcome_timer = 0
        
        while welcome_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_e:
                        # Enter track editor
                        road.clear_track()
                        if not enter_track_editor_mode(screen, road, small_font):
                            pygame.quit()
                            return
                        welcome_active = False
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        # Use default track
                        road.create_default_track()
                        welcome_active = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
            
            # Draw welcome screen
            screen.fill(GREEN)
            
            # Title
            title_text = font.render("AI Racing - Custom Track Mode", True, WHITE)
            title_rect = title_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 150))
            screen.blit(title_text, title_rect)
            
            # Instructions
            instructions = [
                "No custom track detected!",
                "",
                "Options:",
                "E - Create custom track now",
                "SPACE - Use default oval track", 
                "ESC - Quit",
                "",
                "After creating a track, the AI cars will race on it!"
            ]
            
            y_offset = HEIGHT//2 - 80
            for instruction in instructions:
                if instruction:
                    color = YELLOW if instruction.startswith("E -") or instruction.startswith("SPACE -") else WHITE
                    text = small_font.render(instruction, True, color)
                    text_rect = text.get_rect(center=(WIDTH//2, y_offset))
                    screen.blit(text, text_rect)
                y_offset += 30
            
            pygame.display.flip()
            clock.tick(60)
    
    # Set up path tracker with track boundaries
    path_tracker.set_track_boundaries(road)
    
    # Generate cars at appropriate starting position
    cars = generate_cars(num_cars, road)
    generation = apply_brain_to_cars(cars)
    
    # Load previous tracking data
    weight_tracker.load_data()
    decision_tracker.load_data()
    sensor_analyzer.load_data()
    path_tracker.load_data()
    
    # Load mutation tracking data
    mutation_data = load_mutation_data()
    print(f"Loaded {len(mutation_data)} generations of mutation data")
    print("All trackers initialized.")
    print("Controls: H=Heatmaps, D=Decisions, Z=Sensors, M=Mutation, P=Paths, E=Edit Track")

    net_screen = pygame.Surface((net_width, HEIGHT))
    net_screen.fill(BLACK)

    road_screen = pygame.Surface((WIDTH, HEIGHT))
    road_screen.fill(BLACK)

    visualizer = Visualizer()

    running = True
    frame_count = 0
    paused = False
    show_help = False
    
    while running:
        frame_count += 1
        
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
                    
                elif event.key == pygame.K_p:
                    paused = not paused
                    print(f"Game {'paused' if paused else 'resumed'}")
                    
                elif event.key == pygame.K_F1:
                    show_help = not show_help
                    
                elif event.key == pygame.K_e:
                    # Enter track editor mode
                    print("Entering track editor mode...")
                    if not enter_track_editor_mode(screen, road, small_font):
                        if best_car:
                            save_state(best_car, generation)
                        running = False
                    else:
                        # Track was modified, regenerate cars at new starting position
                        path_tracker.set_track_boundaries(road)
                        cars = generate_cars(num_cars, road)
                        apply_brain_to_cars(cars)
                        print("Track updated! Cars regenerated at new starting position.")
                        
                elif event.key == pygame.K_r:
                    if best_car:
                        save_state(best_car, generation)
                    reset_game()
                    
                elif event.key == pygame.K_s:
                    if best_car:
                        save_state(best_car, generation)
                    # Also save the current track
                    road.save_track()
                    
                elif event.key == pygame.K_l:
                    # Load a different track
                    if road.load_track():
                        path_tracker.set_track_boundaries(road)
                        cars = generate_cars(num_cars, road)
                        apply_brain_to_cars(cars)
                        print("New track loaded! Cars regenerated.")
                        
                # Analysis hotkeys
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
                
                elif event.key == pygame.K_o:  # Shift+P for path analysis
                    print("Generating path tracking visualizations...")
                    try:
                        path_tracker.save_data()
                        
                        print("Creating ghost racing visualization...")
                        path_tracker.create_ghost_racing_visualization()
                        
                        print("Creating heat trail visualization...")
                        path_tracker.create_heat_trail_visualization()
                        
                        print("Creating performance overlay...")
                        path_tracker.create_performance_overlay_visualization()
                        
                        print("Creating evolution comparison...")
                        path_tracker.create_evolution_comparison()
                        
                        stats = path_tracker.create_summary_statistics()
                        
                        print("\nAll path visualizations saved successfully!")
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

            # Handle car controls (if you have manual control)
            for car in cars:
                car.controls.handle_event(event)

        if not paused:
            # Update cars
            for car in cars:
                car.update(screen, road.borders)

            # Find best car
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

        # Draw everything
        screen.fill(BLACK)
        
        # Draw road
        road_screen.fill(GREEN)
        road.draw(road_screen)
        screen.blit(road_screen, (0, 0))

        # Draw cars
        for car in cars:
            car.update(screen, road.borders) if not paused else None
            
        if best_car:
            best_car.draw(screen, True)

        # Draw network visualization
        net_screen.fill(BLACK)
        visualizer.update()
        screen.blit(net_screen, (1200, 0))

        # Draw UI
        # Generation info
        gen_text = font.render(f"Generation: {generation}", True, WHITE)
        screen.blit(gen_text, (20, 20))
        
        # Mutation rate
        mut_text = font.render(f"Mutation Rate: {current_mutation_rate:.3f}", True, (255, 200, 100))
        screen.blit(mut_text, (20, 60))
        
        # Best car distance
        if best_car:
            dist_text = font.render(f"Best Distance: {best_car.distance_traveled:.1f}", True, (100, 255, 100))
            screen.blit(dist_text, (20, 100))
        
        # Track info
        track_type = "Custom" if road.drawing_complete and len(road.centerline_points) > 10 else "Default"
        track_text = small_font.render(f"Track: {track_type}", True, (200, 200, 255))
        screen.blit(track_text, (20, 140))
        
        # Tracking stats
        y_offset = 180
        if len(weight_tracker.weight_history) > 0:
            weight_text = small_font.render(f"Weights tracked: {len(weight_tracker.weight_history)} gens", True, (200, 200, 200))
            screen.blit(weight_text, (20, y_offset))
            y_offset += 25
        
        if len(decision_tracker.decision_history) > 0:
            decision_text = small_font.render(f"Decisions tracked: {len(decision_tracker.decision_history)} gens", True, (200, 200, 200))
            screen.blit(decision_text, (20, y_offset))
            y_offset += 25
        
        if len(sensor_analyzer.sensor_history) > 0:
            total_samples = sum(len(data) for data in sensor_analyzer.sensor_history.values())
            sensor_text = small_font.render(f"Sensor samples: {total_samples}", True, (200, 200, 200))
            screen.blit(sensor_text, (20, y_offset))
            y_offset += 25
        
        if len(mutation_data) > 0:
            mutation_text = small_font.render(f"Mutation data: {len(mutation_data)} gens", True, (200, 200, 200))
            screen.blit(mutation_text, (20, y_offset))
            y_offset += 25
        
        if len(path_tracker.path_history) > 0:
            path_text = small_font.render(f"Paths tracked: {len(path_tracker.path_history)} gens", True, (200, 200, 200))
            screen.blit(path_text, (20, y_offset))
            y_offset += 25
        
        # Controls help
        if show_help:
            help_lines = [
                "=== CONTROLS ===",
                "R - Reset Generation",
                "S - Save State & Track", 
                "L - Load Track",
                "E - Edit Track",
                "P - Pause/Resume",
                "1/2/3 - Mutation Rate",
                "",
                "=== ANALYSIS ===", 
                "H - Weight Heatmaps",
                "D - Decision Patterns",
                "Z - Sensor Analysis",
                "M - Mutation Impact",
                "Shift+P - Path Analysis",
                "",
                "F1 - Toggle This Help",
                "ESC - Quit"
            ]
            
            # Semi-transparent background
            help_width = 250
            help_height = len(help_lines) * 20 + 20
            help_surface = pygame.Surface((help_width, help_height))
            help_surface.set_alpha(200)
            help_surface.fill(BLACK)
            screen.blit(help_surface, (WIDTH - help_width - 20, 20))
            
            # Help text
            for i, line in enumerate(help_lines):
                color = YELLOW if line.startswith("===") else WHITE
                help_text = small_font.render(line, True, color)
                screen.blit(help_text, (WIDTH - help_width - 10, 30 + i * 20))
        else:
            # Compact instructions
            instructions_text = small_font.render(
                "F1=Help, E=Edit Track, R=Reset, H/D/Z/M=Analysis, Shift+P=Paths", 
                True, (150, 150, 150)
            )
            screen.blit(instructions_text, (20, y_offset + 10))
        
        # Pause indicator
        if paused:
            pause_text = font.render("PAUSED - Press P to Resume", True, YELLOW)
            pause_rect = pause_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            
            # Semi-transparent background
            overlay = pygame.Surface((pause_rect.width + 40, pause_rect.height + 20))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            overlay_rect = overlay.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(overlay, overlay_rect)
            
            screen.blit(pause_text, pause_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    # Add import for math module at the top
    import math
    main()