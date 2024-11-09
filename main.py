import pygame
import pygame_gui
import pygame._sdl2
import numpy as np
from scipy.ndimage import convolve
import cv2
import sys

# Constants
BLACK = (0, 0, 0)
GREEN = (0, 0, 0)
WHITE = (200, 200, 200)
PANEL_RADIUS = 20

class Game:
    def __init__(self):
        self.window_height = 600
        self.window_width = 800
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.board_size = (100, 100)
        self.board = np.zeros(self.board_size)
        self.board = np.random.choice([0, 1], self.board_size)

        self.block_size = 30
        self.gap = 1
        self.zoom = 2
        self.zoom_limit = (0.1, 25)
        self.moved = False
        
        ## Game Panel
        self.game_panel_margins = (20, 20)
        self.game_panel_size = (int(self.window_width*0.7) - self.game_panel_margins[0]*2,
                                        self.window_height - self.game_panel_margins[1]*2)
        self.game_panel = pygame.Surface(self.game_panel_size)
        self.game_panel_grid = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        self.game_panel_mask = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        pygame.draw.rect(self.game_panel_mask, (255, 255, 255, 0), (0,0)+self.game_panel_size, border_radius=PANEL_RADIUS)
        
        ## Control Panel
        self.manager = pygame_gui.UIManager((self.window_width, self.window_height), theme_path='themes/panel.json')
        self.control_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((int(self.window_width*0.7), self.game_panel_margins[1]), (int(self.window_width*0.3) - self.game_panel_margins[0], self.window_height - self.game_panel_margins[1]*2)),
            starting_height=0,
            manager=self.manager
        )
        self.button_on_panel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(0, 0, 150, 50),
            text='STOP',
            manager=self.manager,
            container=self.control_panel,
            anchors={'center': 'center'}
        )
        self.label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(0, 0, 150, 50),
            text="Control Panel",
            manager=self.manager,
            container=self.control_panel,
            anchors={'centerx': 'centerx'}
        )

        self.last_draw = pygame.time.get_ticks()
        self.next_frame_time = 500
        self.stop_grid = False
        self.running = True       

    def draw_game_panel(self):
        # Background Color
        self.game_panel.fill(GREEN)

        # Draw Game of Life Grid
        if pygame.time.get_ticks() - self.last_draw > self.next_frame_time and not self.stop_grid:
            # self.board = np.random.choice([0, 1], self.board_size)
            self.game_of_life_generation()
            self.draw_board()
            self.resize_board()
            self.last_draw = pygame.time.get_ticks()

        elif self.moved:
            self.resize_board()
            self.moved = False

        #Mask Surface
        self.game_panel_center = (self.game_panel_size[0]//2, self.game_panel_size[1]//2)
        self.game_panel_grid_center = (self.game_panel_grid.get_width()//2, self.game_panel_grid.get_height()//2)
        game_panel_grid_pos = (self.game_panel_center[0] - self.game_panel_grid_center[0], self.game_panel_center[1] - self.game_panel_grid_center[1])
        self.game_panel.blit(self.game_panel_grid, game_panel_grid_pos)
        self.game_panel.blit(self.game_panel_mask, (0,0), special_flags=pygame.BLEND_RGBA_MIN)
        self.screen.blit(self.game_panel, self.game_panel_margins)

    def game_of_life_generation(self):
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        
        neighbors = convolve(self.board, kernel, mode='constant', cval=0)

        self.board = (neighbors == 3) | ((self.board == 1) & (neighbors == 2)).astype(np.uint8)

    def draw_board(self):
        # Constants
        board_size = self.board.shape[0] * self.block_size

        # Draw Grid
        scaled = self.board*255
        self.board_img = cv2.resize(scaled.astype(np.uint8), (board_size, board_size), interpolation=cv2.INTER_NEAREST)
        self.board_img = cv2.cvtColor(self.board_img, cv2.COLOR_GRAY2BGR)
        for i in range(0, board_size, self.block_size):
            cv2.line(self.board_img, (i, 0), (i, board_size), GREEN, self.gap)
            cv2.line(self.board_img, (0, i), (board_size, i), GREEN, self.gap)
        
    def resize_board(self):
        # Constants
        board_size = self.board.shape[0] * self.block_size

        # Calculate the panel dimensions and center coordinates
        panel_width, panel_height = self.game_panel.get_width(), self.game_panel.get_height()
        center_x, center_y = board_size // 2, board_size // 2

        # Calculate the cropping region based on the panel size
        start_x = max(0, center_x - int(panel_width // (2 * self.zoom)))
        start_y = max(0, center_y - int(panel_height // (2 * self.zoom)))
        end_x = min(board_size, center_x + int(panel_width // (2 * self.zoom)))
        end_y = min(board_size, center_y + int(panel_height // (2 * self.zoom)))

        # Crop board image
        cropped_board_img = self.board_img[start_x:end_x, start_y:end_y]

        # Apply zoom to the cropped region
        zoomed_size = (int(cropped_board_img.shape[1] * self.zoom), int(cropped_board_img.shape[0] * self.zoom))
        board_img = cv2.resize(cropped_board_img, zoomed_size, interpolation=cv2.INTER_NEAREST)

        # Convert to Pygame surface
        self.game_panel_grid = pygame.surfarray.make_surface(board_img)
        # cv2.imshow('Image with self.self.gaps', cv2.transpose(board_img))

    def resize_window(self, new_width, new_height):

        self.window_height = new_height
        self.window_width = new_width

        # Resize Game Panel
        self.game_panel_size = (int(self.window_width*0.7) - self.game_panel_margins[0]*2,
                                        self.window_height - self.game_panel_margins[1]*2)
        self.game_panel = pygame.Surface(self.game_panel_size)
        self.game_panel_mask = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        pygame.draw.rect(self.game_panel_mask, (255, 255, 255, 0), (0,0)+self.game_panel_size, border_radius=PANEL_RADIUS)
        
        # Resize Control Panel
        self.manager.set_window_resolution((self.window_width, self.window_height))
        self.control_panel.set_relative_position((int(self.window_width*0.7), self.game_panel_margins[1]))
        self.control_panel.set_dimensions((int(self.window_width*0.3) - self.game_panel_margins[0], self.window_height - self.game_panel_margins[1]*2))
        self.control_panel.rebuild()

    def handle_events(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                self.resize_window(event.w, event.h)

            elif event.type == pygame.MOUSEWHEEL:
                self.moved = True
                if event.y > 0: 
                    self.zoom = min(self.zoom_limit[1], self.zoom*1.2)
                else: 
                    self.zoom = max(self.zoom_limit[0], self.zoom/1.2)
                print(self.zoom)

            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_on_panel:
                    self.stop_grid = not self.stop_grid
            
            self.manager.process_events(event)

    def run(self):
        while self.running:
            self.time_delta = self.clock.tick() / 1000.0

            self.handle_events()
            
            self.draw_game_panel()

            self.manager.update(self.time_delta)
            self.manager.draw_ui(self.screen)
            # print(self.clock.get_fps())
            pygame.display.flip()

        pygame.quit()
        sys.exit()

# Main function
if __name__ == "__main__":
    pygame.init()
    game = Game()
    game.run()
