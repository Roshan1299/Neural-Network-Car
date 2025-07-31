import pygame
import sys
import os

# Add the current directory to path so we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import settings (create a simple one if it doesn't exist)
try:
    from settings import WIDTH, HEIGHT, FPS, WHITE, BLACK, GREEN
except ImportError:
    # Fallback settings if settings.py doesn't exist
    WIDTH, HEIGHT = 1200, 900
    FPS = 60
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (34, 139, 34)

# Import our updated Road class
from environment.road import Road

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Racing Track Editor - Create Your Custom Track!")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 48)
    
    # Create road editor (don't load saved track initially)
    road = Road(load_saved=False)
    road.clear_track()  # Start with blank canvas
    
    # Show welcome message
    show_welcome = True
    welcome_timer = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle road drawing events
            road.handle_event(event)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    show_welcome = False
        
        # Fill background with grass color
        screen.fill(GREEN)
        
        # Draw road
        road.draw(screen)
        
        # Draw instructions
        road.draw_instructions(screen, font)
        
        # Show welcome message for first few seconds
        welcome_timer += clock.get_time()
        if show_welcome and welcome_timer < 5000:  # Show for 5 seconds
            # Semi-transparent overlay
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))
            
            # Welcome text
            title_text = title_font.render("Welcome to Track Editor!", True, WHITE)
            title_rect = title_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 100))
            screen.blit(title_text, title_rect)
            
            instructions = [
                "Create your custom racing track by drawing with your mouse!",
                "",
                "Quick Start:",
                "1. Click and drag to draw your track centerline", 
                "2. Make a complete loop",
                "3. Click once more to finish the track",
                "4. Press 'S' to save your track",
                "5. Run main.py to race on your custom track!",
                "",
                "Press SPACE to dismiss this message"
            ]
            
            y_offset = HEIGHT//2 - 30
            for instruction in instructions:
                if instruction:  # Skip empty lines
                    text = font.render(instruction, True, WHITE)
                    text_rect = text.get_rect(center=(WIDTH//2, y_offset))
                    screen.blit(text, text_rect)
                y_offset += 30
        else:
            show_welcome = False
        
        # Show track status in corner
        status_text = f"Track Status: {'Drawing...' if road.is_drawing else 'Complete' if road.drawing_complete else 'Ready'}"
        status_surface = font.render(status_text, True, WHITE)
        screen.blit(status_surface, (20, HEIGHT - 30))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Ask if user wants to save before quitting
    if road.drawing_complete:
        print("\nTrack editor closing...")
        print("Make sure you saved your track with 'S' key!")
        print("Your saved track will be automatically loaded in main.py")
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()