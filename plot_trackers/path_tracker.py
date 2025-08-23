import pygame
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import defaultdict
from scipy import interpolate
from scipy.ndimage import gaussian_filter

class PathTracker:
    def __init__(self):
        self.path_history = {}  # generation -> list of (x, y, timestamp, speed, steering)
        self.track_boundaries = None  # Store track shape for visualization
        self.generation_stats = {}  # Store performance metrics per generation
        self.sampling_rate = 3  # Sample every N frames to reduce data size
        self.frame_counter = 0
        
    def set_track_boundaries(self, road):
        """Store the track boundaries for visualization reference"""
        self.track_boundaries = {
            'inner_points': road.inner_points.copy(),
            'outer_points': road.outer_points.copy(),
            'center': road.center
        }
    
    def sample_path(self, car, generation, frame_count):
        """Sample car position and metrics during driving"""
        self.frame_counter += 1
        
        # Only sample every N frames to keep data manageable
        if self.frame_counter % self.sampling_rate != 0:
            return
            
        if not car or not car.moving:
            return
            
        if generation not in self.path_history:
            self.path_history[generation] = []
            
        # Calculate additional metrics
        speed = abs(car.speed) if hasattr(car, 'speed') else 0
        steering = 0
        if hasattr(car, 'controls'):
            if car.controls.left:
                steering = -1
            elif car.controls.right:
                steering = 1
                
        # Store path point with metadata
        path_point = {
            'x': car.x,
            'y': car.y,
            'timestamp': frame_count,
            'speed': speed,
            'steering': steering,
            'distance_traveled': car.distance_traveled,
            'angle': car.angle if hasattr(car, 'angle') else 0
        }
        
        self.path_history[generation].append(path_point)
    
    def finalize_generation(self, generation, final_distance, crashed=False):
        """Store final generation statistics"""
        if generation in self.path_history:
            path_data = self.path_history[generation]
            if path_data:
                self.generation_stats[generation] = {
                    'final_distance': final_distance,
                    'path_length': len(path_data),
                    'crashed': crashed,
                    'avg_speed': np.mean([p['speed'] for p in path_data]),
                    'max_speed': max([p['speed'] for p in path_data]),
                    'steering_smoothness': self._calculate_steering_smoothness(path_data),
                    'track_efficiency': self._calculate_track_efficiency(path_data)
                }
    
    def _calculate_steering_smoothness(self, path_data):
        """Calculate how smooth the steering was (less jittery = better)"""
        if len(path_data) < 3:
            return 0
            
        steering_changes = []
        for i in range(1, len(path_data)):
            steering_changes.append(abs(path_data[i]['steering'] - path_data[i-1]['steering']))
        
        # Lower values = smoother steering
        return np.mean(steering_changes)
    
    def _calculate_track_efficiency(self, path_data):
        """Calculate how efficiently the car used the track width"""
        if not self.track_boundaries or len(path_data) < 10:
            return 0.5  # Default middle value
            
        # This is a simplified calculation - in practice you'd want more sophisticated geometry
        center_x, center_y = self.track_boundaries['center']
        distances_from_center = []
        
        for point in path_data:
            dist = math.sqrt((point['x'] - center_x)**2 + (point['y'] - center_y)**2)
            distances_from_center.append(dist)
        
        # Normalize distance variation (lower = more consistent racing line)
        return np.std(distances_from_center) / np.mean(distances_from_center) if distances_from_center else 0.5
    
    def save_data(self):
        """Save path tracking data to file"""
        data = {
            'path_history': self.path_history,
            'generation_stats': self.generation_stats,
            'track_boundaries': self.track_boundaries
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean data for JSON serialization
        clean_data = json.loads(json.dumps(data, default=convert_numpy))
        
        with open('path_tracking_data.json', 'w') as f:
            json.dump(clean_data, f, indent=2)
    
    def load_data(self):
        """Load existing path tracking data"""
        if os.path.exists('path_tracking_data.json'):
            with open('path_tracking_data.json', 'r') as f:
                data = json.load(f)
                self.path_history = {int(k): v for k, v in data.get('path_history', {}).items()}
                self.generation_stats = {int(k): v for k, v in data.get('generation_stats', {}).items()}
                self.track_boundaries = data.get('track_boundaries')
                print(f"Loaded path data for {len(self.path_history)} generations")
    
    def create_ghost_racing_visualization(self):
        """Create visualization showing evolution of racing lines"""
        if len(self.path_history) < 2:
            print("Need at least 2 generations for ghost racing visualization")
            return
            
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw track boundaries
        if self.track_boundaries:
            inner_points = self.track_boundaries['inner_points']
            outer_points = self.track_boundaries['outer_points']
            
            # Convert to arrays for plotting
            inner_x, inner_y = zip(*inner_points)
            outer_x, outer_y = zip(*outer_points)
            
            ax.plot(inner_x + (inner_x[0],), inner_y + (inner_y[0],), 
                   color='lime', linewidth=3, label='Track Inner Boundary')
            ax.plot(outer_x + (outer_x[0],), outer_y + (outer_y[0],), 
                   color='lime', linewidth=3, label='Track Outer Boundary')
            
            # Fill track area
            track_x = list(outer_x) + list(reversed(inner_x))
            track_y = list(outer_y) + list(reversed(inner_y))
            ax.fill(track_x, track_y, color='gray', alpha=0.3, label='Track Surface')
        
        # Plot paths for different generations with ghost effect
        generations = sorted(self.path_history.keys())
        colors = plt.cm.plasma(np.linspace(0, 1, len(generations)))
        
        for i, gen in enumerate(generations):
            path_data = self.path_history[gen]
            if len(path_data) < 10:  # Skip very short paths
                continue
                
            x_coords = [p['x'] for p in path_data]
            y_coords = [p['y'] for p in path_data]
            
            # Alpha decreases for older generations (ghost effect)
            alpha = 0.3 + 0.7 * (i / len(generations))
            linewidth = 1 + 2 * (i / len(generations))
            
            ax.plot(x_coords, y_coords, color=colors[i], alpha=alpha, 
                   linewidth=linewidth, label=f'Generation {gen}')
        
        # Highlight the best and worst performers
        if generations:
            best_gen = max(generations, key=lambda g: self.generation_stats.get(g, {}).get('final_distance', 0))
            worst_gen = min(generations, key=lambda g: self.generation_stats.get(g, {}).get('final_distance', float('inf')))
            
            # Re-plot best in bright color
            if best_gen in self.path_history:
                best_path = self.path_history[best_gen]
                best_x = [p['x'] for p in best_path]
                best_y = [p['y'] for p in best_path]
                ax.plot(best_x, best_y, color='gold', linewidth=4, alpha=0.9, 
                       label=f'Best (Gen {best_gen})', zorder=10)
        
        ax.set_title('AI Car Path Evolution: Ghost Racing Visualization', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig('path_evolution_ghost_racing.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_heat_trail_visualization(self):
        """Create heatmap showing most frequently used racing lines"""
        if not self.path_history:
            print("No path data available for heat trail visualization")
            return
            
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Collect all path points
        all_x, all_y = [], []
        for gen_data in self.path_history.values():
            all_x.extend([p['x'] for p in gen_data])
            all_y.extend([p['y'] for p in gen_data])
        
        if not all_x:
            print("No path points found")
            return
        
        # Create 2D histogram for heat map
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Create grid for heatmap
        bins = 100
        x_edges = np.linspace(x_min, x_max, bins)
        y_edges = np.linspace(y_min, y_max, bins)
        
        H, _, _ = np.histogram2d(all_x, all_y, bins=[x_edges, y_edges])
        H = gaussian_filter(H, sigma=1.5)  # Smooth the heatmap
        
        # Create custom colormap
        colors = ['black', 'purple', 'red', 'orange', 'yellow', 'white']
        custom_cmap = LinearSegmentedColormap.from_list('heat_trail', colors)
        
        # Plot heatmap
        extent = [x_min, x_max, y_min, y_max]
        im = ax.imshow(H.T, origin='lower', extent=extent, cmap=custom_cmap, alpha=0.8)
        
        # Draw track boundaries on top
        if self.track_boundaries:
            inner_points = self.track_boundaries['inner_points']
            outer_points = self.track_boundaries['outer_points']
            
            inner_x, inner_y = zip(*inner_points)
            outer_x, outer_y = zip(*outer_points)
            
            ax.plot(inner_x + (inner_x[0],), inner_y + (inner_y[0],), 
                   color='lime', linewidth=3, alpha=0.7, label='Track Boundaries')
            ax.plot(outer_x + (outer_x[0],), outer_y + (outer_y[0],), 
                   color='lime', linewidth=3, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Path Usage Frequency', fontsize=12)
        
        ax.set_title('AI Car Heat Trail: Most Used Racing Lines', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig('path_heat_trail.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_overlay_visualization(self):
        """Create visualization with speed and steering overlays"""
        if not self.path_history:
            print("No path data available for performance overlay")
            return
            
        # Get the best performing generation
        best_gen = max(self.path_history.keys(), 
                      key=lambda g: self.generation_stats.get(g, {}).get('final_distance', 0))
        
        best_path = self.path_history[best_gen]
        if len(best_path) < 10:
            print("Not enough data points for performance overlay")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('black')
        
        # Extract coordinates and metrics
        x_coords = [p['x'] for p in best_path]
        y_coords = [p['y'] for p in best_path]
        speeds = [p['speed'] for p in best_path]
        steering = [abs(p['steering']) for p in best_path]
        
        # Speed visualization
        scatter1 = ax1.scatter(x_coords, y_coords, c=speeds, cmap='plasma', 
                             s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Draw track boundaries
        if self.track_boundaries:
            inner_points = self.track_boundaries['inner_points']
            outer_points = self.track_boundaries['outer_points']
            
            inner_x, inner_y = zip(*inner_points)
            outer_x, outer_y = zip(*outer_points)
            
            ax1.plot(inner_x + (inner_x[0],), inner_y + (inner_y[0],), 
                    color='lime', linewidth=2, alpha=0.6)
            ax1.plot(outer_x + (outer_x[0],), outer_y + (outer_y[0],), 
                    color='lime', linewidth=2, alpha=0.6)
            ax2.plot(inner_x + (inner_x[0],), inner_y + (inner_y[0],), 
                    color='lime', linewidth=2, alpha=0.6)
            ax2.plot(outer_x + (outer_x[0],), outer_y + (outer_y[0],), 
                    color='lime', linewidth=2, alpha=0.6)
        
        # Steering visualization
        scatter2 = ax2.scatter(x_coords, y_coords, c=steering, cmap='coolwarm', 
                             s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Formatting
        for ax in [ax1, ax2]:
            ax.set_facecolor('black')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, color='gray')
        
        ax1.set_title(f'Speed Analysis - Generation {best_gen}\\nBest Distance: {self.generation_stats[best_gen]["final_distance"]:.1f}', 
                     fontsize=14, fontweight='bold', color='white')
        ax2.set_title(f'Steering Analysis - Generation {best_gen}\\nSteering Smoothness: {self.generation_stats[best_gen]["steering_smoothness"]:.3f}', 
                     fontsize=14, fontweight='bold', color='white')
        
        # Add colorbars
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Speed', fontsize=12, color='white')
        cbar1.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Steering Intensity', fontsize=12, color='white')
        cbar2.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
        
        plt.tight_layout()
        plt.savefig('path_performance_overlay.png', dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.show()
    
    def create_evolution_comparison(self):
        """Create side-by-side comparison of early vs late generation paths"""
        if len(self.path_history) < 10:
            print("Need at least 10 generations for evolution comparison")
            return
            
        generations = sorted(self.path_history.keys())
        early_gens = generations[:3]  # First 3 generations
        late_gens = generations[-3:]  # Last 3 generations
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('black')
        
        # Plot early generations
        colors_early = plt.cm.Reds(np.linspace(0.3, 0.8, len(early_gens)))
        for i, gen in enumerate(early_gens):
            if gen in self.path_history:
                path_data = self.path_history[gen]
                x_coords = [p['x'] for p in path_data]
                y_coords = [p['y'] for p in path_data]
                ax1.plot(x_coords, y_coords, color=colors_early[i], 
                        linewidth=2, alpha=0.8, label=f'Gen {gen}')
        
        # Plot late generations
        colors_late = plt.cm.Blues(np.linspace(0.3, 0.8, len(late_gens)))
        for i, gen in enumerate(late_gens):
            if gen in self.path_history:
                path_data = self.path_history[gen]
                x_coords = [p['x'] for p in path_data]
                y_coords = [p['y'] for p in path_data]
                ax2.plot(x_coords, y_coords, color=colors_late[i], 
                        linewidth=2, alpha=0.8, label=f'Gen {gen}')
        
        # Draw track boundaries on both
        if self.track_boundaries:
            inner_points = self.track_boundaries['inner_points']
            outer_points = self.track_boundaries['outer_points']
            
            inner_x, inner_y = zip(*inner_points)
            outer_x, outer_y = zip(*outer_points)
            
            for ax in [ax1, ax2]:
                ax.plot(inner_x + (inner_x[0],), inner_y + (inner_y[0],), 
                       color='lime', linewidth=3, alpha=0.7)
                ax.plot(outer_x + (outer_x[0],), outer_y + (outer_y[0],), 
                       color='lime', linewidth=3, alpha=0.7)
        
        # Formatting
        for ax in [ax1, ax2]:
            ax.set_facecolor('black')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, color='gray')
            ax.legend()
        
        ax1.set_title('Early Generations\\n(Learning Phase)', fontsize=14, fontweight='bold', color='white')
        ax2.set_title('Late Generations\\n(Mastery Phase)', fontsize=14, fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig('path_evolution_comparison.png', dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.show()
    
    def create_summary_statistics(self):
        """Generate summary statistics and insights"""
        if not self.generation_stats:
            print("No generation statistics available")
            return
            
        print("\\n=== PATH TRACKING ANALYSIS SUMMARY ===")
        
        generations = sorted(self.generation_stats.keys())
        
        # Distance progression
        distances = [self.generation_stats[g]['final_distance'] for g in generations]
        print(f"Distance Evolution:")
        print(f"  First Generation: {distances[0]:.1f}")
        print(f"  Last Generation: {distances[-1]:.1f}")
        print(f"  Best Ever: {max(distances):.1f} (Gen {generations[distances.index(max(distances))]})") 
        print(f"  Improvement: {distances[-1] - distances[0]:.1f} ({((distances[-1]/distances[0] - 1) * 100):.1f}% increase)")
        
        # Speed analysis
        avg_speeds = [self.generation_stats[g]['avg_speed'] for g in generations if 'avg_speed' in self.generation_stats[g]]
        if avg_speeds:
            print(f"\\nSpeed Evolution:")
            print(f"  Early Average Speed: {np.mean(avg_speeds[:5]):.3f}")
            print(f"  Late Average Speed: {np.mean(avg_speeds[-5:]):.3f}")
            print(f"  Speed Improvement: {np.mean(avg_speeds[-5:]) - np.mean(avg_speeds[:5]):.3f}")
        
        # Steering smoothness
        smoothness = [self.generation_stats[g]['steering_smoothness'] for g in generations if 'steering_smoothness' in self.generation_stats[g]]
        if smoothness:
            print(f"\\nSteering Evolution:")
            print(f"  Early Smoothness: {np.mean(smoothness[:5]):.3f} (higher = more erratic)")
            print(f"  Late Smoothness: {np.mean(smoothness[-5:]):.3f}")
            print(f"  Smoothness Improvement: {np.mean(smoothness[:5]) - np.mean(smoothness[-5:]):.3f}")
        
        return {
            'distance_improvement': distances[-1] - distances[0] if distances else 0,
            'best_generation': generations[distances.index(max(distances))] if distances else 0,
            'best_distance': max(distances) if distances else 0,
            'total_generations': len(generations)
        }