import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import defaultdict, deque
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime

class SensorAnalyzer:
    def __init__(self, max_history=1000):
        self.sensor_history = defaultdict(list)  # generation -> sensor data
        self.decision_history = defaultdict(list)  # generation -> decisions
        self.survival_history = defaultdict(list)  # generation -> survival times
        self.critical_moments = defaultdict(list)  # generation -> critical moment data
        self.max_history = max_history
        
        # Sensor names for better visualization
        self.sensor_names = [
            'Left Forward', 'Left', 'Center', 'Right', 'Right Forward'
        ]
        self.action_names = ['Forward', 'Left', 'Right', 'Reverse']
        
        # Real-time tracking
        self.current_frame_data = deque(maxlen=100)
        
    def sample_sensor_data(self, car, generation, frame_count=0):
        """Sample sensor data and decisions from a car"""
        if not car or not car.sensor or not car.brain:
            return
            
        # Get sensor readings (normalized distances)
        sensor_readings = []
        for reading in car.sensor.readings:
            if reading is not None:
                # Convert offset to distance (1.0 = no obstacle, 0.0 = immediate obstacle)
                distance = reading["offset"]
                sensor_readings.append(distance)
            else:
                sensor_readings.append(1.0)  # No obstacle detected
        
        # Ensure we have 5 sensors
        while len(sensor_readings) < 5:
            sensor_readings.append(1.0)
        sensor_readings = sensor_readings[:5]
        
        # Get current neural network outputs
        try:
            inputs = car.brain.preprocess_inputs(sensor_readings)
            outputs = car.brain.feed_forward(inputs, car.brain)
        except:
            outputs = [0, 0, 0, 0]
        
        # Calculate sensor influence (weight magnitudes)
        sensor_influence = self._calculate_sensor_influence(car.brain, sensor_readings)
        
        # Detect critical moments (close to obstacles)
        is_critical = any(reading < 0.3 for reading in sensor_readings)
        
        # Store the data
        data_point = {
            'timestamp': frame_count,
            'sensor_readings': sensor_readings,
            'outputs': outputs,
            'sensor_influence': sensor_influence,
            'survival_time': car.distance_traveled,
            'speed': car.speed,
            'is_critical': is_critical,
            'position': (car.x, car.y),
            'angle': car.angle
        }
        
        self.sensor_history[generation].append(data_point)
        
        # Track critical moments separately
        if is_critical:
            self.critical_moments[generation].append(data_point)
            
        # Real-time tracking
        self.current_frame_data.append(data_point)
        
    def _calculate_sensor_influence(self, brain, sensor_readings):
        """Calculate how much each sensor influences the final decision"""
        influences = []
        
        if not brain.levels:
            return [0] * len(sensor_readings)
            
        # Sum absolute weights from each input to all first hidden layer neurons
        first_level = brain.levels[0]
        
        for sensor_idx in range(len(sensor_readings)):
            if sensor_idx < len(first_level.weights):
                # Sum absolute weights for this sensor across all outputs
                influence = sum(abs(weight) for weight in first_level.weights[sensor_idx])
                influences.append(influence)
            else:
                influences.append(0)
                
        return influences
    
    def create_sensor_importance_chart(self):
        """Create bar chart showing sensor importance across all generations"""
        if not self.sensor_history:
            print("No sensor data available")
            return
            
        # Aggregate data across all generations
        all_influences = [[] for _ in range(5)]
        all_correlations = [[] for _ in range(5)]
        
        for generation, data_points in self.sensor_history.items():
            if not data_points:
                continue
                
            # Calculate average influence for each sensor in this generation
            gen_influences = [[] for _ in range(5)]
            survival_times = []
            sensor_readings_by_sensor = [[] for _ in range(5)]
            
            for point in data_points:
                for i, influence in enumerate(point['sensor_influence']):
                    gen_influences[i].append(influence)
                    
                survival_times.append(point['survival_time'])
                for i, reading in enumerate(point['sensor_readings']):
                    sensor_readings_by_sensor[i].append(reading)
            
            # Average influence per sensor for this generation
            for i in range(5):
                if gen_influences[i]:
                    avg_influence = np.mean(gen_influences[i])
                    all_influences[i].append(avg_influence)
                    
                    # Calculate correlation with survival time
                    if len(sensor_readings_by_sensor[i]) > 1 and len(survival_times) > 1:
                        corr, _ = pearsonr(sensor_readings_by_sensor[i], survival_times)
                        all_correlations[i].append(abs(corr) if not np.isnan(corr) else 0)
                    else:
                        all_correlations[i].append(0)
        
        # Create the visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sensor Utilization Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average Sensor Influence (Weight Magnitude)
        avg_influences = [np.mean(influences) if influences else 0 for influences in all_influences]
        std_influences = [np.std(influences) if influences else 0 for influences in all_influences]
        
        bars1 = ax1.bar(self.sensor_names, avg_influences, yerr=std_influences, 
                       capsize=5, color='skyblue', alpha=0.8)
        ax1.set_title('Sensor Importance (Weight Magnitude)', fontweight='bold')
        ax1.set_ylabel('Average Influence')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars1, avg_influences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 2. Correlation with Survival
        avg_correlations = [np.mean(corrs) if corrs else 0 for corrs in all_correlations]
        
        colors = ['red' if corr < 0.1 else 'orange' if corr < 0.3 else 'green' 
                 for corr in avg_correlations]
        bars2 = ax2.bar(self.sensor_names, avg_correlations, color=colors, alpha=0.8)
        ax2.set_title('Sensor-Survival Correlation', fontweight='bold')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars2, avg_correlations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 3. Sensor Activity Heatmap (across generations)
        if len(self.sensor_history) > 1:
            activity_matrix = []
            generations = sorted(self.sensor_history.keys())
            
            for gen in generations:
                if self.sensor_history[gen]:
                    avg_readings = [0] * 5
                    for point in self.sensor_history[gen]:
                        for i, reading in enumerate(point['sensor_readings']):
                            avg_readings[i] += (1.0 - reading)  # Invert: higher = more obstacle detection
                    
                    # Normalize by number of data points
                    avg_readings = [r / len(self.sensor_history[gen]) for r in avg_readings]
                    activity_matrix.append(avg_readings)
            
            if activity_matrix:
                sns.heatmap(np.array(activity_matrix).T, 
                           xticklabels=[f'Gen {g}' for g in generations],
                           yticklabels=self.sensor_names,
                           cmap='YlOrRd', annot=True, fmt='.2f', ax=ax3)
                ax3.set_title('Sensor Activity Across Generations', fontweight='bold')
                ax3.set_xlabel('Generation')
        
        # 4. Critical Moment Analysis
        critical_sensor_usage = [0] * 5
        total_critical_moments = 0
        
        for generation, critical_data in self.critical_moments.items():
            for moment in critical_data:
                total_critical_moments += 1
                for i, influence in enumerate(moment['sensor_influence']):
                    critical_sensor_usage[i] += influence
        
        if total_critical_moments > 0:
            critical_sensor_usage = [usage / total_critical_moments for usage in critical_sensor_usage]
            
            bars4 = ax4.bar(self.sensor_names, critical_sensor_usage, 
                           color='red', alpha=0.7)
            ax4.set_title('Sensor Usage in Critical Moments', fontweight='bold')
            ax4.set_ylabel('Average Influence')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars4, critical_sensor_usage):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'sensor_importance_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sensor importance analysis saved as sensor_importance_analysis_{timestamp}.png")
    
    def create_real_time_sensor_display(self):
        """Create real-time sensor readings visualization"""
        if not self.current_frame_data:
            print("No real-time data available")
            return
            
        # Get the most recent data
        latest_data = self.current_frame_data[-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Real-Time Sensor Analysis', fontsize=16, fontweight='bold')
        
        # 1. Current sensor readings
        readings = latest_data['sensor_readings']
        influences = latest_data['sensor_influence']
        
        # Color code by influence
        colors = plt.cm.viridis([inf / max(influences) if max(influences) > 0 else 0 
                                for inf in influences])
        
        bars1 = ax1.bar(self.sensor_names, readings, color=colors, alpha=0.8)
        ax1.set_title('Current Sensor Readings', fontweight='bold')
        ax1.set_ylabel('Distance to Obstacle (1.0 = far, 0.0 = close)')
        ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add influence as text on bars
        for bar, reading, influence in zip(bars1, readings, influences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{reading:.2f}\n({influence:.2f})', ha='center', va='bottom')
        
        # 2. Recent history (last 50 frames)
        if len(self.current_frame_data) > 10:
            history_length = min(50, len(self.current_frame_data))
            recent_data = list(self.current_frame_data)[-history_length:]
            
            # Plot sensor readings over time
            for sensor_idx in range(5):
                sensor_values = [data['sensor_readings'][sensor_idx] for data in recent_data]
                ax2.plot(sensor_values, label=self.sensor_names[sensor_idx], 
                        linewidth=2, alpha=0.8)
            
            ax2.set_title('Recent Sensor History', fontweight='bold')
            ax2.set_ylabel('Distance to Obstacle')
            ax2.set_xlabel('Frames Ago')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'real_time_sensors_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Real-time sensor display saved as real_time_sensors_{timestamp}.png")
    
    def create_sensor_redundancy_analysis(self):
        """Analyze which sensors provide similar information"""
        if not self.sensor_history:
            print("No sensor data available")
            return
            
        # Collect all sensor readings
        all_sensor_data = [[] for _ in range(5)]
        
        for generation, data_points in self.sensor_history.items():
            for point in data_points:
                for i, reading in enumerate(point['sensor_readings']):
                    all_sensor_data[i].append(reading)
        
        # Calculate correlation matrix between sensors
        sensor_df = pd.DataFrame({
            name: data for name, data in zip(self.sensor_names, all_sensor_data)
        })
        
        correlation_matrix = sensor_df.corr()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Sensor Redundancy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Inter-Sensor Correlation', fontweight='bold')
        
        # 2. Redundancy analysis
        redundancy_scores = []
        for i in range(5):
            # For each sensor, find its maximum correlation with other sensors
            correlations = [abs(correlation_matrix.iloc[i, j]) 
                           for j in range(5) if i != j]
            max_correlation = max(correlations) if correlations else 0
            redundancy_scores.append(max_correlation)
        
        colors = ['red' if score > 0.7 else 'orange' if score > 0.5 else 'green' 
                 for score in redundancy_scores]
        
        bars = ax2.bar(self.sensor_names, redundancy_scores, color=colors, alpha=0.8)
        ax2.set_title('Sensor Redundancy Scores', fontweight='bold')
        ax2.set_ylabel('Max Correlation with Other Sensors')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Redundancy')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Redundancy')
        ax2.legend()
        
        # Add value labels
        for bar, val in zip(bars, redundancy_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'sensor_redundancy_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sensor redundancy analysis saved as sensor_redundancy_analysis_{timestamp}.png")
    
    def get_sensor_insights(self):
        """Generate textual insights about sensor usage"""
        if not self.sensor_history:
            return "No sensor data available for analysis."
        
        insights = []
        insights.append("=== SENSOR UTILIZATION INSIGHTS ===\n")
        
        # Calculate overall statistics
        total_samples = sum(len(data) for data in self.sensor_history.values())
        generations_tracked = len(self.sensor_history)
        critical_moments_total = sum(len(data) for data in self.critical_moments.values())
        
        insights.append(f"📊 Data Overview:")
        insights.append(f"   • Total samples: {total_samples}")
        insights.append(f"   • Generations tracked: {generations_tracked}")
        insights.append(f"   • Critical moments: {critical_moments_total}")
        
        # Analyze sensor importance
        all_influences = [[] for _ in range(5)]
        for generation, data_points in self.sensor_history.items():
            for point in data_points:
                for i, influence in enumerate(point['sensor_influence']):
                    all_influences[i].append(influence)
        
        avg_influences = [np.mean(influences) if influences else 0 for influences in all_influences]
        most_important = np.argmax(avg_influences)
        least_important = np.argmin(avg_influences)
        
        insights.append(f"\n🎯 Sensor Importance:")
        insights.append(f"   • Most important: {self.sensor_names[most_important]} ({avg_influences[most_important]:.3f})")
        insights.append(f"   • Least important: {self.sensor_names[least_important]} ({avg_influences[least_important]:.3f})")
        
        # Analyze critical moments
        if critical_moments_total > 0:
            critical_rate = (critical_moments_total / total_samples) * 100
            insights.append(f"\n⚠️  Critical Moments Analysis:")
            insights.append(f"   • Critical moment rate: {critical_rate:.1f}%")
            
            # Which sensors are most active during critical moments
            critical_influences = [0] * 5
            for generation, critical_data in self.critical_moments.items():
                for moment in critical_data:
                    for i, influence in enumerate(moment['sensor_influence']):
                        critical_influences[i] += influence
            
            if sum(critical_influences) > 0:
                critical_influences = [inf / sum(critical_influences) for inf in critical_influences]
                most_critical = np.argmax(critical_influences)
                insights.append(f"   • Most active in crises: {self.sensor_names[most_critical]} ({critical_influences[most_critical]*100:.1f}%)")
        
        # Evolution insights
        if generations_tracked > 5:
            early_gens = sorted(self.sensor_history.keys())[:3]
            late_gens = sorted(self.sensor_history.keys())[-3:]
            
            early_influences = [[] for _ in range(5)]
            late_influences = [[] for _ in range(5)]
            
            for gen in early_gens:
                for point in self.sensor_history[gen]:
                    for i, influence in enumerate(point['sensor_influence']):
                        early_influences[i].append(influence)
            
            for gen in late_gens:
                for point in self.sensor_history[gen]:
                    for i, influence in enumerate(point['sensor_influence']):
                        late_influences[i].append(influence)
            
            early_avg = [np.mean(inf) if inf else 0 for inf in early_influences]
            late_avg = [np.mean(inf) if inf else 0 for inf in late_influences]
            
            changes = [late - early for late, early in zip(late_avg, early_avg)]
            biggest_change = np.argmax([abs(change) for change in changes])
            
            insights.append(f"\n📈 Evolution Insights:")
            insights.append(f"   • Biggest change: {self.sensor_names[biggest_change]}")
            insights.append(f"   • Change magnitude: {changes[biggest_change]:+.3f}")
            
            if changes[biggest_change] > 0:
                insights.append(f"   • Trend: AI learned to rely more on this sensor")
            else:
                insights.append(f"   • Trend: AI reduced dependence on this sensor")
        
        return "\n".join(insights)
    
    def save_data(self, filename="sensor_analysis_data.json"):
        """Save sensor analysis data to file"""
        data = {
            'sensor_history': dict(self.sensor_history),
            'critical_moments': dict(self.critical_moments),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Sensor analysis data saved to {filename}")
    
    def load_data(self, filename="sensor_analysis_data.json"):
        """Load sensor analysis data from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.sensor_history = defaultdict(list, {
                    int(k): v for k, v in data.get('sensor_history', {}).items()
                })
                self.critical_moments = defaultdict(list, {
                    int(k): v for k, v in data.get('critical_moments', {}).items()
                })
                
                print(f"Loaded sensor analysis data from {filename}")
                print(f"Generations tracked: {len(self.sensor_history)}")
                return True
            except Exception as e:
                print(f"Error loading sensor data: {e}")
                return False
        return False