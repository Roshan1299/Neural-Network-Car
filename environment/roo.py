import pygame
import math
import json
import os
from settings import *

class Road:
    def __init__(self, load_saved=True):
        # Track drawing properties
        self._num_segments = 50
        self.track_width = 80
        self.centerline_points = []
        self.inner_points = []
        self.outer_points = []
        self.borders = []
        self.is_drawing = False
        self.drawing_complete = False
        self.min_point_distance = 15
        
        # Compatibility attributes for existing code
        self.center = (WIDTH // 2, HEIGHT // 2)
        self.inner_radius = 180
        self.outer_radius = 320
        self.num_segments = 50
        self.stretch_x = 1.5
        self.wavy_amplitude = 120
        self.wavy_frequency = 2
        
        # Try to load saved track first, otherwise create default
        if load_saved and not self.load_track():
            self.create_default_track()
        elif not load_saved:
            self.create_default_track()

    def create_default_track(self):
        """Create a simple default oval track (fallback)"""
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        self.center = (center_x, center_y)  # Update center for compatibility
        
        # Create an oval track similar to your original wavy track
        self.centerline_points = []
        num_points = 40
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            # Create an oval shape with some stretch
            x = center_x + 200 * math.cos(angle) * 1.5
            y = center_y + 150 * math.sin(angle)
            self.centerline_points.append((x, y))
        
        self.create_track_boundaries()
        self.drawing_complete = True
        print("Using default oval track")

    def start_drawing(self, pos):
        """Start drawing a new track"""
        self.is_drawing = True
        self.drawing_complete = False
        self.centerline_points = [pos]
        self.inner_points = []
        self.outer_points = []
        self.borders = []
        print("Started drawing track. Click and drag to draw the centerline!")

    def add_point(self, pos):
        """Add a point to the track while drawing"""
        if not self.is_drawing:
            return
            
        # Only add point if it's far enough from the last point
        if len(self.centerline_points) > 0:
            last_point = self.centerline_points[-1]
            distance = math.sqrt((pos[0] - last_point[0])**2 + (pos[1] - last_point[1])**2)
            if distance < self.min_point_distance:
                return
        
        self.centerline_points.append(pos)
        
        # Create boundaries in real-time for preview
        if len(self.centerline_points) > 2:
            self.create_track_boundaries()

    def finish_drawing(self):
        """Finish drawing and close the track loop"""
        if not self.is_drawing or len(self.centerline_points) < 4:
            print("Need at least 4 points to create a track!")
            return False
            
        # Close the loop by connecting to start
        start_point = self.centerline_points[0]
        last_point = self.centerline_points[-1]
        distance_to_start = math.sqrt((start_point[0] - last_point[0])**2 + 
                                    (start_point[1] - last_point[1])**2)
        
        # If not close to start, add intermediate points
        if distance_to_start > 50:
            # Add a smooth curve back to start
            steps = int(distance_to_start / 30)
            for i in range(1, steps + 1):
                t = i / (steps + 1)
                x = last_point[0] + t * (start_point[0] - last_point[0])
                y = last_point[1] + t * (start_point[1] - last_point[1])
                self.centerline_points.append((x, y))
        
        self.is_drawing = False
        self.drawing_complete = True
        self.create_track_boundaries()
        print(f"Track completed with {len(self.centerline_points)} points!")
        return True

    def create_track_boundaries(self):
        """Create inner and outer track boundaries from centerline"""
        if len(self.centerline_points) < 3:
            return
            
        self.inner_points = []
        self.outer_points = []
        
        for i in range(len(self.centerline_points)):
            curr_point = self.centerline_points[i]
            
            # Get direction vector (tangent)
            if i == 0:
                next_point = self.centerline_points[1]
                direction = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            elif i == len(self.centerline_points) - 1:
                prev_point = self.centerline_points[-2]
                direction = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            else:
                prev_point = self.centerline_points[i-1]
                next_point = self.centerline_points[i+1]
                direction = (next_point[0] - prev_point[0], next_point[1] - prev_point[1])
            
            # Normalize direction
            length = math.sqrt(direction[0]**2 + direction[1]**2)
            if length > 0:
                direction = (direction[0]/length, direction[1]/length)
            else:
                direction = (1, 0)
            
            # Get perpendicular vector (normal)
            normal = (-direction[1], direction[0])
            
            # Calculate inner and outer points
            half_width = self.track_width / 2
            inner_x = curr_point[0] + normal[0] * half_width
            inner_y = curr_point[1] + normal[1] * half_width
            outer_x = curr_point[0] - normal[0] * half_width
            outer_y = curr_point[1] - normal[1] * half_width
            
            self.inner_points.append((inner_x, inner_y))
            self.outer_points.append((outer_x, outer_y))
        
        # Update num_segments for compatibility
        self.num_segments = len(self.inner_points)
        
        self.create_borders()

    def create_borders(self):
        """Create border line segments for collision detection"""
        self.borders = []
        
        if len(self.inner_points) < 2 or len(self.outer_points) < 2:
            return
            
        # Inner boundary segments
        for i in range(len(self.inner_points)):
            start = self.inner_points[i]
            end = self.inner_points[(i + 1) % len(self.inner_points)]
            self.borders.append((start, end))
        
        # Outer boundary segments
        for i in range(len(self.outer_points)):
            start = self.outer_points[i]
            end = self.outer_points[(i + 1) % len(self.outer_points)]
            self.borders.append((start, end))

    def clear_track(self):
        """Clear the current track"""
        self.centerline_points = []
        self.inner_points = []
        self.outer_points = []
        self.borders = []
        self.is_drawing = False
        self.drawing_complete = False
        print("Track cleared. Click to start drawing a new track!")

    def save_track(self, filename="custom_track.json"):
        """Save the current track to a file"""
        if not self.drawing_complete:
            print("Cannot save incomplete track!")
            return False
            
        track_data = {
            "centerline_points": self.centerline_points,
            "track_width": self.track_width
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(track_data, f, indent=2)
            print(f"Track saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving track: {e}")
            return False

    def load_track(self, filename="custom_track.json"):
        """Load a track from a file"""
        try:
            if not os.path.exists(filename):
                print(f"Track file {filename} not found")
                return False
                
            with open(filename, 'r') as f:
                track_data = json.load(f)
            
            self.centerline_points = track_data["centerline_points"]
            self.track_width = track_data.get("track_width", 80)
            
            # Calculate track center from centerline points for compatibility
            if self.centerline_points:
                avg_x = sum(p[0] for p in self.centerline_points) / len(self.centerline_points)
                avg_y = sum(p[1] for p in self.centerline_points) / len(self.centerline_points)
                self.center = (avg_x, avg_y)
            
            self.create_track_boundaries()
            self.drawing_complete = True
            self.is_drawing = False
            
            print(f"Track loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading track: {e}")
            return False

    def handle_event(self, event):
        """Handle pygame events for drawing"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if not self.is_drawing and not self.drawing_complete:
                    # Start drawing
                    self.start_drawing(event.pos)
                elif self.is_drawing:
                    # Finish drawing
                    self.finish_drawing()
            elif event.button == 3:  # Right click
                if self.is_drawing:
                    # Cancel current drawing
                    self.clear_track()
                
        elif event.type == pygame.MOUSEMOTION:
            if self.is_drawing and pygame.mouse.get_pressed()[0]:
                # Add point while dragging
                self.add_point(event.pos)
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # Clear track
                self.clear_track()
            elif event.key == pygame.K_s:
                # Save track
                self.save_track()
            elif event.key == pygame.K_l:
                # Load track
                self.load_track()
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                # Increase track width
                self.track_width = min(150, self.track_width + 10)
                if self.drawing_complete:
                    self.create_track_boundaries()
                print(f"Track width: {self.track_width}")
            elif event.key == pygame.K_MINUS:
                # Decrease track width
                self.track_width = max(40, self.track_width - 10)
                if self.drawing_complete:
                    self.create_track_boundaries()
                print(f"Track width: {self.track_width}")

    def draw(self, screen):
        """Draw the track - compatible with your original drawing style"""
        # Draw track surface (similar to your original style)
        if len(self.inner_points) > 2 and len(self.outer_points) > 2 and self.drawing_complete:
            # Draw track surface (gray)
            all_points = self.outer_points + self.inner_points[::-1]
            pygame.draw.polygon(screen, (60, 60, 60), all_points)
            
            # Draw inner area (grass/black)
            pygame.draw.polygon(screen, BLACK, self.inner_points, 0)
            
            # Draw track boundaries (grass colored borders like your original)
            pygame.draw.polygon(screen, (34, 139, 34), self.inner_points, 4)
            pygame.draw.polygon(screen, (34, 139, 34), self.outer_points, 4)
        
        # Drawing mode visuals
        if self.is_drawing:
            # Draw centerline while drawing
            if len(self.centerline_points) > 1:
                pygame.draw.lines(screen, (255, 255, 0), False, self.centerline_points, 2)
            
            # Draw current drawing point
            if len(self.centerline_points) > 0:
                pygame.draw.circle(screen, (255, 0, 0), self.centerline_points[-1], 5)
            
            # Draw preview boundaries while drawing
            if len(self.inner_points) > 1:
                pygame.draw.lines(screen, (100, 100, 100), False, self.inner_points, 2)
                pygame.draw.lines(screen, (100, 100, 100), False, self.outer_points, 2)

    def draw_instructions(self, screen, font):
        """Draw instructions on screen"""
        instructions = [
            "TRACK DRAWING CONTROLS:",
            "Left Click + Drag: Draw track centerline",
            "Left Click: Finish drawing",
            "Right Click: Cancel/Clear",
            "C: Clear track",
            "S: Save track",
            "L: Load track", 
            "+/-: Adjust track width",
            f"Current width: {self.track_width}",
            f"Status: {'Drawing...' if self.is_drawing else 'Complete' if self.drawing_complete else 'Ready to draw'}"
        ]
        
        y_offset = 20
        for i, instruction in enumerate(instructions):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            text = font.render(instruction, True, color)
            screen.blit(text, (WIDTH - 350, y_offset + i * 25))

    # Compatibility methods to match your original Road class interface
    @property
    def num_segments(self):
        """Return number of segments for compatibility"""
        return len(self.inner_points) if self.inner_points else 0
    
    @property
    def num_segments(self):
        return self._num_segments

    @num_segments.setter
    def num_segments(self, value):
        self._num_segments = value
