import pygame
import sys

# Settings (copy from your settings.py)
WIDTH, HEIGHT = 1200, 900
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ASPHALT = (53, 70, 92)

# Import the updated Road class
# (You would replace this with: from road import Road)

import math

class Road:
    def __init__(self):
        self.track_width = 40
        
        # Define the black track from the image - this is a complex circuit
        self.centerline_points = [
            # Start/finish straight (bottom)
            (200, 800), (300, 800), (400, 800), (500, 800), (600, 800),
            
            # First right turn
            (650, 790), (690, 770), (720, 740), (740, 700), (750, 650),
            
            # Going up the right side
            (750, 600), (750, 550), (750, 500), (750, 450),
            
            # Sharp right turn at top-right
            (740, 400), (720, 360), (690, 330), (650, 310), (600, 300),
            
            # Top straight going left
            (550, 300), (500, 300), (450, 300), (400, 300), (350, 300),
            
            # Left turn going down
            (300, 310), (260, 330), (230, 360), (210, 400), (200, 450),
            
            # Down the left side
            (200, 500), (200, 550), (200, 600),
            
            # Complex middle section - hairpin and curves
            (210, 640), (230, 670), (260, 690), (300, 700), (350, 700),
            (400, 700), (450, 690), (490, 670), (520, 640), (540, 600),
            
            # Tight hairpin turn
            (550, 560), (560, 520), (570, 480), (580, 440), (590, 400),
            (600, 360), (610, 320), (620, 280), (630, 240), (640, 200),
            
            # Coming back around the hairpin
            (650, 160), (670, 130), (700, 110), (740, 100), (780, 100),
            (820, 110), (850, 130), (870, 160), (880, 200), (870, 240),
            (850, 280), (820, 320), (780, 360), (740, 400),
            
            # Back through middle section
            (700, 440), (660, 480), (620, 520), (580, 560), (540, 600),
            (500, 640), (460, 680), (420, 700), (380, 710), (340, 710),
            
            # Final curves back to start
            (300, 700), (260, 690), (230, 670), (210, 640), (200, 600),
            (200, 650), (200, 700), (200, 750),
            
            # Back to start/finish
            (210, 790), (230, 800), (250, 800)
        ]
        
        # Generate inner and outer boundaries
        self.inner_points, self.outer_points = self.create_track_boundaries()
        self.borders = self.create_borders()



    def create_track_boundaries(self):
        """Create inner and outer track boundaries from centerline"""
        inner_points = []
        outer_points = []
        
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
            
            inner_points.append((inner_x, inner_y))
            outer_points.append((outer_x, outer_y))
        
        return inner_points, outer_points

    def create_borders(self):
        """Create border line segments for collision detection"""
        borders = []
        
        # Inner boundary segments
        for i in range(len(self.inner_points)):
            start = self.inner_points[i]
            end = self.inner_points[(i + 1) % len(self.inner_points)]
            borders.append((start, end))
        
        # Outer boundary segments
        for i in range(len(self.outer_points)):
            start = self.outer_points[i]
            end = self.outer_points[(i + 1) % len(self.outer_points)]
            borders.append((start, end))
        
        return borders

    def draw(self, screen):
        """Draw the track"""
        # Draw track surface (dark gray/black asphalt)
        if len(self.inner_points) > 2 and len(self.outer_points) > 2:
            # Create a combined polygon for the track surface
            all_points = self.outer_points + self.inner_points[::-1]
            pygame.draw.polygon(screen, (40, 40, 40), all_points)
        
        # Draw track boundaries (white lines like in the image)
        if len(self.inner_points) > 1:
            pygame.draw.lines(screen, WHITE, True, self.inner_points, 3)
        if len(self.outer_points) > 1:
            pygame.draw.lines(screen, WHITE, True, self.outer_points, 3)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Road Test")
    clock = pygame.time.Clock()
    
    # Create road
    road = Road()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Fill background
        screen.fill((34, 139, 34))  # Green grass color
        
        # Draw road
        road.draw(screen)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()