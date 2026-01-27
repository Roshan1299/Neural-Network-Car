# decision_tracker.py
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
import math

class DecisionTracker:
    def __init__(self):
        self.decision_history = {}  # {generation: decision_data}
        self.frame_counter = 0
        self.sampling_interval = 10  # Sample every N frames
        
        # Decision categories for radar chart
        self.categories = [
            'Aggressive Forward',
            'Gentle Forward',
            'Sharp Left',
            'Gentle Left', 
            'Sharp Right',
            'Gentle Right',
            'Reverse/Brake',
            'Decision Confidence',
            'Reaction Speed'
        ]
        
    def sample_decision(self, car, generation, sensor_readings):
        """Sample and categorize a car's decision"""
        self.frame_counter += 1
        
        if self.frame_counter % self.sampling_interval != 0:
            return
            
        if generation not in self.decision_history:
            self.decision_history[generation] = {
                'decisions': [],
                'sensor_history': [],
                'previous_outputs': None,
                'frame_count': 0
            }
        
        # Get current outputs from the car's brain
        processed_inputs = car.brain.preprocess_inputs(sensor_readings)
        outputs = car.brain.feed_forward(processed_inputs, car.brain)
        
        # Categorize the decision
        decision_data = self._categorize_decision(
            outputs, 
            sensor_readings,
            self.decision_history[generation]['previous_outputs']
        )
        
        self.decision_history[generation]['decisions'].append(decision_data)
        self.decision_history[generation]['sensor_history'].append(sensor_readings)
        self.decision_history[generation]['previous_outputs'] = outputs
        self.decision_history[generation]['frame_count'] += 1
    
    def _categorize_decision(self, outputs, sensors, previous_outputs):
        """Categorize a decision into behavioral patterns"""
        # Assuming outputs are [forward, left, right, reverse] based on your network
        forward, left, right, reverse = outputs
        
        decision = {
            'aggressive_forward': 0,
            'gentle_forward': 0,
            'sharp_left': 0,
            'gentle_left': 0,
            'sharp_right': 0,
            'gentle_right': 0,
            'reverse_brake': 0,
            'decision_confidence': 0,
            'reaction_speed': 0
        }
        
        # Forward movement categorization
        if forward > 0.8:
            decision['aggressive_forward'] = 1
        elif forward > 0.3:
            decision['gentle_forward'] = 1
            
        # Steering categorization
        if left > 0.8:
            decision['sharp_left'] = 1
        elif left > 0.3:
            decision['gentle_left'] = 1
            
        if right > 0.8:
            decision['sharp_right'] = 1
        elif right > 0.3:
            decision['gentle_right'] = 1
            
        # Reverse/brake
        if reverse > 0.3:
            decision['reverse_brake'] = 1
            
        # Decision confidence (how decisive are the outputs?)
        max_output = max(outputs)
        min_output = min(outputs)
        decision['decision_confidence'] = max_output - min_output
        
        # Reaction speed (how much did outputs change from previous?)
        if previous_outputs:
            output_change = sum(abs(curr - prev) for curr, prev in zip(outputs, previous_outputs))
            decision['reaction_speed'] = min(output_change, 1.0)  # Cap at 1.0
        
        return decision
    
    def calculate_generation_stats(self, generation):
        """Calculate statistics for a generation's decision patterns"""
        if generation not in self.decision_history:
            return None
            
        decisions = self.decision_history[generation]['decisions']
        if not decisions:
            return None
            
        stats = {}
        
        # Calculate frequencies and averages
        total_decisions = len(decisions)
        
        stats['aggressive_forward'] = sum(d['aggressive_forward'] for d in decisions) / total_decisions
        stats['gentle_forward'] = sum(d['gentle_forward'] for d in decisions) / total_decisions
        stats['sharp_left'] = sum(d['sharp_left'] for d in decisions) / total_decisions
        stats['gentle_left'] = sum(d['gentle_left'] for d in decisions) / total_decisions
        stats['sharp_right'] = sum(d['sharp_right'] for d in decisions) / total_decisions
        stats['gentle_right'] = sum(d['gentle_right'] for d in decisions) / total_decisions
        stats['reverse_brake'] = sum(d['reverse_brake'] for d in decisions) / total_decisions
        
        # Average confidence and reaction speed
        stats['decision_confidence'] = sum(d['decision_confidence'] for d in decisions) / total_decisions
        stats['reaction_speed'] = sum(d['reaction_speed'] for d in decisions) / total_decisions
        
        return stats
    
    def create_radar_chart(self, generations=None, save_path="decision_patterns_radar.png", figsize=(12, 10)):
        """Create radar chart comparing decision patterns across generations"""
        if generations is None:
            generations = sorted(list(self.decision_history.keys()))
            
        if len(generations) == 0:
            print("No generation data available for radar chart")
            return
            
        # Limit to most interesting generations for readability
        if len(generations) > 6:
            # Show early, middle, and late generations
            step = len(generations) // 5
            generations = generations[::step][:6]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Number of categories
        N = len(self.categories)
        
        # Angles for each category
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Colors for different generations
        colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
        
        # Plot each generation
        for i, generation in enumerate(generations):
            stats = self.calculate_generation_stats(generation)
            if not stats:
                continue
                
            # Extract values for radar chart
            values = [
                stats['aggressive_forward'],
                stats['gentle_forward'], 
                stats['sharp_left'],
                stats['gentle_left'],
                stats['sharp_right'],
                stats['gentle_right'],
                stats['reverse_brake'],
                stats['decision_confidence'],
                stats['reaction_speed']
            ]
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Generation {generation}', 
                   color=colors[i], alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)
        
        plt.title('AI Car Decision Making Patterns Evolution', size=16, pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_evolution_timeline(self, save_path="decision_evolution_timeline.png"):
        """Create a timeline showing how decision patterns change over generations"""
        generations = sorted(list(self.decision_history.keys()))
        if len(generations) < 2:
            print("Need at least 2 generations for timeline")
            return
            
        # Prepare data
        timeline_data = {category: [] for category in ['aggressive_forward', 'gentle_forward', 
                                                      'sharp_left', 'sharp_right', 'reverse_brake',
                                                      'decision_confidence']}
        
        for gen in generations:
            stats = self.calculate_generation_stats(gen)
            if stats:
                for category in timeline_data.keys():
                    timeline_data[category].append(stats[category])
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Action frequencies
        ax1.plot(generations, timeline_data['aggressive_forward'], 'r-', linewidth=2, 
                label='Aggressive Forward', marker='o')
        ax1.plot(generations, timeline_data['gentle_forward'], 'orange', linewidth=2, 
                label='Gentle Forward', marker='s')
        ax1.plot(generations, timeline_data['sharp_left'], 'b-', linewidth=2, 
                label='Sharp Left', marker='^')
        ax1.plot(generations, timeline_data['sharp_right'], 'g-', linewidth=2, 
                label='Sharp Right', marker='v')
        ax1.plot(generations, timeline_data['reverse_brake'], 'purple', linewidth=2, 
                label='Reverse/Brake', marker='x')
        
        ax1.set_title('Decision Pattern Evolution: Action Frequencies', fontsize=14)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Decision quality metrics
        ax2.plot(generations, timeline_data['decision_confidence'], 'darkred', linewidth=3, 
                label='Decision Confidence', marker='o')
        
        ax2.set_title('Decision Pattern Evolution: Quality Metrics', fontsize=14)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Confidence Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_behavioral_comparison(self, early_gen=None, late_gen=None, save_path="behavioral_comparison.png"):
        """Create side-by-side comparison of early vs late generation behavior"""
        generations = sorted(list(self.decision_history.keys()))
        
        if early_gen is None:
            early_gen = generations[0] if generations else None
        if late_gen is None:
            late_gen = generations[-1] if generations else None
            
        if not early_gen or not late_gen or early_gen == late_gen:
            print("Need at least 2 different generations for comparison")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
        
        # Angles for radar chart
        N = len(self.categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        
        # Early generation
        early_stats = self.calculate_generation_stats(early_gen)
        if early_stats:
            early_values = [
                early_stats['aggressive_forward'], early_stats['gentle_forward'],
                early_stats['sharp_left'], early_stats['gentle_left'],
                early_stats['sharp_right'], early_stats['gentle_right'],
                early_stats['reverse_brake'], early_stats['decision_confidence'],
                early_stats['reaction_speed']
            ]
            early_values += early_values[:1]
            
            ax1.plot(angles, early_values, 'o-', linewidth=3, color='red', alpha=0.8)
            ax1.fill(angles, early_values, alpha=0.2, color='red')
            ax1.set_title(f'Early Generation {early_gen}\n(Learning Phase)', size=14, pad=20)
        
        # Late generation  
        late_stats = self.calculate_generation_stats(late_gen)
        if late_stats:
            late_values = [
                late_stats['aggressive_forward'], late_stats['gentle_forward'],
                late_stats['sharp_left'], late_stats['gentle_left'],
                late_stats['sharp_right'], late_stats['gentle_right'],
                late_stats['reverse_brake'], late_stats['decision_confidence'],
                late_stats['reaction_speed']
            ]
            late_values += late_values[:1]
            
            ax2.plot(angles, late_values, 'o-', linewidth=3, color='blue', alpha=0.8)
            ax2.fill(angles, late_values, alpha=0.2, color='blue')
            ax2.set_title(f'Late Generation {late_gen}\n(Evolved Behavior)', size=14, pad=20)
        
        # Customize both charts
        for ax in [ax1, ax2]:
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.categories, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.grid(True)
        
        plt.suptitle('AI Behavioral Evolution: Early vs Late Generation', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_insights(self):
        """Generate text insights about the behavioral evolution"""
        generations = sorted(list(self.decision_history.keys()))
        if len(generations) < 2:
            return "Need more generation data for insights"
            
        early_gen = generations[0]
        late_gen = generations[-1]
        
        early_stats = self.calculate_generation_stats(early_gen)
        late_stats = self.calculate_generation_stats(late_gen)
        
        if not early_stats or not late_stats:
            return "Insufficient data for insights"
            
        insights = []
        
        # Aggression analysis
        early_aggression = early_stats['aggressive_forward'] + early_stats['sharp_left'] + early_stats['sharp_right']
        late_aggression = late_stats['aggressive_forward'] + late_stats['sharp_left'] + late_stats['sharp_right']
        
        if late_aggression > early_aggression:
            insights.append("🚗 AI became MORE aggressive - learned to drive faster and take sharper turns")
        else:
            insights.append("🚗 AI became more cautious - learned smoother, safer driving patterns")
            
        # Confidence analysis
        confidence_change = late_stats['decision_confidence'] - early_stats['decision_confidence']
        if confidence_change > 0.1:
            insights.append(f"🧠 Decision confidence increased by {confidence_change:.2f} - AI became more decisive")
        elif confidence_change < -0.1:
            insights.append(f"🧠 Decision confidence decreased by {abs(confidence_change):.2f} - AI learned to be more nuanced")
        
        # Reverse usage analysis
        reverse_change = early_stats['reverse_brake'] - late_stats['reverse_brake']
        if reverse_change > 0.1:
            insights.append(f"✅ Reverse usage decreased by {reverse_change:.2f} - AI learned to avoid getting stuck")
            
        return "\n".join(insights)
    
    def save_data(self, filename="decision_tracking_data.json"):
        """Save tracking data to file"""
        with open(filename, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
    
    def load_data(self, filename="decision_tracking_data.json"):
        """Load tracking data from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.decision_history = json.load(f)
            return True
        return False


# Integration instructions
def integration_instructions():
    """
    Integration steps for your main.py:
    
    1. Add this import:
       from decision_tracker import DecisionTracker
    
    2. Initialize in main():
       decision_tracker = DecisionTracker()
       decision_tracker.load_data()
    
    3. In your main game loop, after car.update():
       if best_car and best_car.moving:
           # Get sensor readings (you'll need to expose this from your car)
           sensor_readings = [reading.distance for reading in best_car.sensors]
           decision_tracker.sample_decision(best_car, generation, sensor_readings)
    
    4. Add keyboard shortcut for generating charts:
       elif event.key == pygame.K_d:  # Press 'D' for decision charts
           decision_tracker.create_radar_chart()
           decision_tracker.create_evolution_timeline()
           decision_tracker.create_behavioral_comparison()
           print(decision_tracker.get_insights())
           decision_tracker.save_data()
    
    5. Save data when generation changes:
       decision_tracker.save_data()
    """
    pass

if __name__ == "__main__":
    print("Decision Making Patterns Tracker")
    print("Follow the integration instructions to add this to your project!")