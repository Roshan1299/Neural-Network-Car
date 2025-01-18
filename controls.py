import pygame

class Controls:
    def __init__(self, control_type):
        self.forward = False
        self.left = False
        self.right = False
        self.reverse = False
        self.type = control_type

        if self.type == "KEYS":
            self._add_keyboard_listeners()

    def _add_keyboard_listeners(self):
        self.key_map = {
            pygame.K_LEFT: "left",
            pygame.K_RIGHT: "right",
            pygame.K_UP: "forward",
            pygame.K_DOWN: "reverse",
        }

    def handle_event(self, event):
        if self.type == "KEYS":
            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                is_pressed = event.type == pygame.KEYDOWN
                if event.key in self.key_map:
                    setattr(self, self.key_map[event.key], is_pressed)
