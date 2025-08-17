# weight_tracker.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import defaultdict

class WeightTracker:
    def __init__(self):
        self.weight_history = []  # List of flattened weight arrays for each generation
        self.generation_labels = []  # Generation numbers
        self.layer_info = []  # Information about layer structure for labeling
        
    def track_weights(self, neural_network, generation):
        """Extract and store weights from the best neural network of current generation"""
        # Extract weights from all levels
        all_weights = []
        layer_info = []
        
        for level_idx, level in enumerate(neural_network.levels):
            # Flatten the weight matrix for this level
            level_weights = []
            for i in range(len(level.weights)):
                for j in range(len(level.weights[i])):
                    level_weights.append(level.weights[i][j])
                    
            all_weights.extend(level_weights)
            
            # Store layer information for labeling
            layer_info.append({
                'level': level_idx,
                'input_size': len(level.weights),
                'output_size': len(level.weights[0]) if level.weights else 0,
                'weight_count': len(level_weights),
                'start_idx': len(all_weights) - len(level_weights),
                'end_idx': len(all_weights)
            })
        
        # Store the weights and generation info
        self.weight_history.append(all_weights)
        self.generation_labels.append(generation)
        
        # Update layer info (only needed once, but updated for safety)
        if not self.layer_info:
            self.layer_info = layer_info
            
    def create_heatmap(self, save_path="weight_evolution_heatmap.png", figsize=(15, 10)):
        """Create and display the weight evolution heatmap"""
        if len(self.weight_history) < 2:
            print("Need at least 2 generations to create meaningful heatmap")
            return
            
        # Convert to numpy array (rows = weights, columns = generations)
        weight_matrix = np.array(self.weight_history).T
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        
        # Use a diverging colormap centered at 0
        ax = sns.heatmap(weight_matrix, 
                        xticklabels=self.generation_labels,
                        cmap='RdBu_r',  # Red-Blue colormap (red=positive, blue=negative)
                        center=0,  # Center the colormap at 0
                        cbar_kws={'label': 'Weight Value'},
                        linewidths=0)
        
        # Add layer separators and labels
        self._add_layer_separators(ax)
        
        plt.title('Neural Network Weight Evolution Across Generations', fontsize=16, pad=20)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Network Connections', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def _add_layer_separators(self, ax):
        """Add horizontal lines to separate different layers in the heatmap"""
        for i, layer in enumerate(self.layer_info[:-1]):  # Don't add line after last layer
            y_pos = layer['end_idx'] - 0.5
            ax.axhline(y=y_pos, color='white', linewidth=2)
            
        # Add layer labels on the right side
        for i, layer in enumerate(self.layer_info):
            y_center = (layer['start_idx'] + layer['end_idx']) / 2
            layer_name = f"Level {layer['level']} ({layer['input_size']}→{layer['output_size']})"
            ax.text(len(self.generation_labels) + 0.5, y_center, layer_name, 
                   rotation=0, ha='left', va='center', fontsize=10)
    
    def create_layer_specific_heatmaps(self, save_dir="layer_heatmaps"):
        """Create separate heatmaps for each layer"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for layer_idx, layer in enumerate(self.layer_info):
            # Extract weights for this specific layer
            layer_weights = []
            for gen_weights in self.weight_history:
                layer_weights.append(gen_weights[layer['start_idx']:layer['end_idx']])
            
            layer_matrix = np.array(layer_weights).T
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(layer_matrix,
                       xticklabels=self.generation_labels,
                       cmap='RdBu_r',
                       center=0,
                       cbar_kws={'label': 'Weight Value'})
            
            plt.title(f'Layer {layer_idx} Weight Evolution ({layer["input_size"]}→{layer["output_size"]})', 
                     fontsize=14)
            plt.xlabel('Generation')
            plt.ylabel('Connection Index')
            plt.xticks(rotation=45)
            
            save_path = os.path.join(save_dir, f'layer_{layer_idx}_evolution.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_weight_change_heatmap(self, save_path="weight_changes_heatmap.png"):
        """Create a heatmap showing weight changes between consecutive generations"""
        if len(self.weight_history) < 2:
            print("Need at least 2 generations to show weight changes")
            return
            
        # Calculate differences between consecutive generations
        weight_changes = []
        change_labels = []
        
        for i in range(1, len(self.weight_history)):
            current_weights = np.array(self.weight_history[i])
            previous_weights = np.array(self.weight_history[i-1])
            weight_change = current_weights - previous_weights
            weight_changes.append(weight_change)
            change_labels.append(f"{self.generation_labels[i-1]}→{self.generation_labels[i]}")
        
        change_matrix = np.array(weight_changes).T
        
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(change_matrix,
                        xticklabels=change_labels,
                        cmap='RdBu_r',
                        center=0,
                        cbar_kws={'label': 'Weight Change'})
        
        self._add_layer_separators(ax)
        
        plt.title('Weight Changes Between Generations', fontsize=16, pad=20)
        plt.xlabel('Generation Transition')
        plt.ylabel('Network Connections')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_statistics(self):
        """Get interesting statistics about weight evolution"""
        if len(self.weight_history) < 2:
            return {}
            
        weight_matrix = np.array(self.weight_history)
        
        stats = {
            'total_connections': len(self.weight_history[0]),
            'generations_tracked': len(self.weight_history),
            'weight_range': {
                'min': float(np.min(weight_matrix)),
                'max': float(np.max(weight_matrix)),
                'mean': float(np.mean(weight_matrix)),
                'std': float(np.std(weight_matrix))
            },
            'most_stable_connections': [],
            'most_volatile_connections': []
        }
        
        # Find most stable and volatile connections
        weight_std = np.std(weight_matrix, axis=0)
        most_stable_idx = np.argmin(weight_std)
        most_volatile_idx = np.argmax(weight_std)
        
        stats['most_stable_connections'] = {
            'index': int(most_stable_idx),
            'std': float(weight_std[most_stable_idx]),
            'values': weight_matrix[:, most_stable_idx].tolist()
        }
        
        stats['most_volatile_connections'] = {
            'index': int(most_volatile_idx),
            'std': float(weight_std[most_volatile_idx]),
            'values': weight_matrix[:, most_volatile_idx].tolist()
        }
        
        return stats
    
    def save_data(self, filename="weight_tracking_data.json"):
        """Save tracking data to file"""
        data = {
            'weight_history': self.weight_history,
            'generation_labels': self.generation_labels,
            'layer_info': self.layer_info
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_data(self, filename="weight_tracking_data.json"):
        """Load tracking data from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.weight_history = data['weight_history']
                self.generation_labels = data['generation_labels']
                self.layer_info = data['layer_info']
            return True
        return False


# Integration code for your main.py
def integrate_weight_tracking():
    """
    Instructions for integrating weight tracking into your main.py:
    
    1. Add this import at the top of main.py:
       from weight_tracker import WeightTracker
    
    2. Initialize the tracker in your main() function:
       weight_tracker = WeightTracker()
       weight_tracker.load_data()  # Load previous tracking data if exists
    
    3. In your save_state() function, add weight tracking:
       def save_state(car, generation):
           brain_state = car.brain.get_state()
           save_data = {
               "generation": generation,
               "brain": brain_state
           }
           with open("save_state.json", "w") as f:
               json.dump(save_data, f)
           
           # Add weight tracking here
           weight_tracker.track_weights(car.brain, generation)
           weight_tracker.save_data()
    
    4. Add keyboard shortcut for generating heatmap (in your event handling):
       elif event.key == pygame.K_h:  # Press 'H' to generate heatmap
           weight_tracker.create_heatmap()
           weight_tracker.create_layer_specific_heatmaps()
           weight_tracker.create_weight_change_heatmap()
           stats = weight_tracker.get_statistics()
           print("Weight Evolution Statistics:")
           print(json.dumps(stats, indent=2))
    """
    pass

if __name__ == "__main__":
    # Example usage
    tracker = WeightTracker()
    
    # Simulate some weight tracking (replace with real neural network)
    print("This is an example of how the WeightTracker works.")
    print("Integrate it into your main.py following the instructions in the integrate_weight_tracking() function.")