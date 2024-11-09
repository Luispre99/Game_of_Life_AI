import pygame
import pygame_gui
import pygame._sdl2
import numpy as np
from scipy.ndimage import convolve
import cv2
import sys
import math

# Constants
BLACK = (0, 0, 0)
GREEN = (50, 50, 50)
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

        self.gap = 1
        self.block_size = 30 + self.gap
        self.zoom = 1
        self.zoom_limit = (0.01, 25)
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
        self.next_frame_time = 100
        self.stop_grid = False
        self.running = True       

    def draw_game_panel(self):
        # Background Color
        self.game_panel.fill(GREEN)

        # Draw Game of Life Grid
        if pygame.time.get_ticks() - self.last_draw > self.next_frame_time and not self.stop_grid:
            # self.board = np.random.choice([0, 1], self.board_size)
            self.game_of_life_generation()
            self.resize_board()
            self.draw_board()
            self.last_draw = pygame.time.get_ticks()

        elif self.moved:
            self.resize_board()
            self.draw_board()
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

    def resize_board(self):
        # Constants
        board_size = self.board.shape

        # Calculate the panel dimensions and the board's center coordinates
        panel_width, panel_height = self.game_panel.get_width(), self.game_panel.get_height()
        center_x, center_y = board_size[0] // 2, board_size[1] // 2

        # Calculate the cropping region based on the panel size
        start_x = max(0, int(center_x - (panel_width // (2 * self.zoom * self.block_size))-1))
        start_y = max(0, int(center_y - (panel_height // (2 * self.zoom * self.block_size))-1))
        end_x = min(board_size[0], int(center_x + (panel_width // (2 * self.zoom * self.block_size))+1))
        end_y = min(board_size[1], int(center_y + (panel_height // (2 * self.zoom * self.block_size))+1))
        print(start_x,start_y, end_x, end_y)
        # Crop board
        self.cropped_board = self.board[start_x:end_x, start_y:end_y]

    def draw_board(self):
        # Draw Grid and apply zoom to the cropped region

        cropped_board = self.cropped_board*255
        zoomed_size = (int(cropped_board.shape[1] * int(self.zoom * self.block_size)),
                       int(cropped_board.shape[0] * int(self.zoom * self.block_size)))
        cropped_img = cv2.resize(cropped_board, zoomed_size, interpolation=cv2.INTER_NEAREST)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
        
        # Draw grid lines
        for i in range(0, cropped_img.shape[1], int(self.block_size * self.zoom)):
            cv2.line(cropped_img, (i, 0), (i, zoomed_size[1]), GREEN, self.gap)
        for i in range(0, cropped_img.shape[0], int(self.block_size * self.zoom)):
            cv2.line(cropped_img, (0, i), (zoomed_size[0], i), GREEN, self.gap)

        # Convert to Pygame surface
        self.game_panel_grid = pygame.surfarray.make_surface(cropped_img)
        # cv2.imshow('Image with self.self.gaps', cv2.transpose(cropped_img))
    
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
