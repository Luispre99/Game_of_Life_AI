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
GAME_WIDTH_RATIO = 0.7
CONTROL_WIDTH_RATIO = 1 - GAME_WIDTH_RATIO

class Game:
    def __init__(self):
        self.window_height = 600
        self.window_width = 800
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        self.screen.fill(GREEN)
        self.clock = pygame.time.Clock()
        
        ## Game Panel
        self.game_panel_margins = (20, 20)
        self.game_panel_size = (int(self.window_width * GAME_WIDTH_RATIO - self.game_panel_margins[0] * 1.5),
                                                      self.window_height - self.game_panel_margins[1] * 2)
        self.game_panel = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        self.game_panel_grid = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        self.game_panel_mask = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        pygame.draw.rect(self.game_panel_mask, (255, 255, 255, 255), (0,0)+self.game_panel_size, border_radius=PANEL_RADIUS)
        
        ## Control Panel
        self.manager = pygame_gui.UIManager((self.window_width, self.window_height), theme_path='themes/panel.json')
        self.control_panel_pos  = (self.game_panel_size[0] + self.game_panel_margins[0] * 2, self.game_panel_margins[1])
        self.control_panel_size = (int(self.window_width * CONTROL_WIDTH_RATIO) - self.game_panel_margins[0] * 1.5,
                                                             self.window_height - self.game_panel_margins[1] * 2)
        self.control_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(self.control_panel_pos,
                                      self.control_panel_size),
            object_id="#control_panel",
            starting_height=0,
            manager=self.manager
        )

        ## Button Panel
        self.button_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(0, 200, self.control_panel_size[0], 150),
            object_id="#button_panel",
            starting_height=2,
            manager=self.manager,
            container=self.control_panel,
            anchors={
                'left': 'left',
                'right': 'right',
                'top': 'top'}
        )
        self.aux_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((0, 0), (0, 200)),
            text='',
            object_id="#stop_button",
            manager=self.manager,
            container=self.button_panel,
            anchors={"center":"center"}
        )
        self.stop_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 10), (self.control_panel_size[0]//2-20, 50)),
            text='STOP',
            object_id="#stop_button",
            manager=self.manager,
            container=self.button_panel,
            anchors={"right": "right",
                     "left": "left",
                     "right_target":self.aux_button}
        )
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 10), (self.control_panel_size[0]//2-20, 50)),
            text='RESET',
            object_id="#reset_button",
            manager=self.manager,
            container=self.button_panel,
            anchors={"right": "right",
                     "left": "left",
                     "left_target":self.aux_button}
        )
        self.mix_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, -60, self.control_panel_size[0]//2-20, 50),
            text='MIX',
            object_id="#mix_button",
            manager=self.manager,
            container=self.button_panel,
            anchors={"bottom": "bottom",
                     "right": "right",
                     "left": "left",
                     "right_target":self.aux_button}
        )
        self.dummy_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, -60, self.control_panel_size[0]//2-20, 50),
            text='DUMB',
            object_id="#dummy_button",
            manager=self.manager,
            container=self.button_panel,
            anchors={"bottom": "bottom",
                     "right": "right",
                     "left": "left",
                     "left_target":self.aux_button}
        )

        ## Slider Panel
        self.slider_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(0, -100, self.control_panel_size[0], 100),
            starting_height=0,
            manager=self.manager,
            container=self.control_panel,
            object_id="#slider_panel",
            anchors={
                'left': 'left',
                'right': 'right',
                'bottom': 'bottom'
            }
        )
        self.slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, 30, self.control_panel_size[0]-20, 20),
            start_value=250,
            value_range=(0, 500),
            manager=self.manager,
            container=self.slider_panel,
            anchors={
                'left': 'left',
                'right': 'right'
            }
        )
        self.slider_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, -20), (50, 20)),
            text='',
            manager=self.manager,
            container=self.slider_panel,
            anchors={ 
                'centerx': 'centerx',
                'bottom': 'bottom',
                'centerx_target': self.slider, 
                'bottom_target': self.slider
            }
        )

        self.board_size = (200, 200)
        self.board = np.zeros(self.board_size)
        self.board = np.random.choice([0, 1], self.board_size)

        self.gap = 1
        self.zoom = self.game_panel_size[0]/self.board_size[0]
        self.zoom_limit = (0.01, 500)
        self.moved = False
        self.last_draw = pygame.time.get_ticks()
        self.next_frame_time = 100
        self.stop_grid = False
        self.running = True       

    def draw_game_panel(self):
        # Background Color
        self.game_panel.fill(WHITE)

        # Draw Game of Life Grid
        if pygame.time.get_ticks() - self.last_draw > self.next_frame_time and not self.stop_grid:
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
        self.game_panel.blit(self.game_panel_mask, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
        pygame.draw.rect(self.game_panel, (255, 255, 255, 255), (0,0)+self.game_panel_size, border_radius=PANEL_RADIUS, width=3)
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
        start_x = max(0, int(center_x - (panel_width // (2 * self.zoom))-1))
        start_y = max(0, int(center_y - (panel_height // (2 * self.zoom))-1))
        end_x = min(board_size[0], int(center_x + (panel_width // (2 * self.zoom))+1))
        end_y = min(board_size[1], int(center_y + (panel_height // (2 * self.zoom))+1))

        # Crop board
        self.cropped_board = self.board[start_x:end_x, start_y:end_y]

    def draw_board(self):
        # Draw Grid and apply zoom to the cropped region
        cropped_board = self.cropped_board*255
        zoomed_size = (int(cropped_board.shape[1] * self.zoom ),
                       int(cropped_board.shape[0] * self.zoom ))
        cropped_img = cv2.resize(cropped_board, zoomed_size, interpolation=cv2.INTER_NEAREST)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

        # Draw grid lines
        n = self.cropped_board.shape[0]
        vertical = np.linspace(0, cropped_img.shape[0], n+1).astype(int)
        for i in vertical:
            cv2.line(cropped_img, (0, i), (zoomed_size[0], i), GREEN, self.gap)
        
        n = self.cropped_board.shape[1]
        horizontal = np.linspace(0, cropped_img.shape[1], n+1).astype(int)
        for i in horizontal:
            cv2.line(cropped_img, (i, 0), (i, zoomed_size[1]), GREEN, self.gap)

        # Convert to Pygame surface
        self.game_panel_grid = pygame.surfarray.make_surface(cropped_img)
        # cv2.imshow('Image with self.self.gaps', cv2.transpose(cropped_img))
    
    def resize_window(self, new_width, new_height):
        self.screen.fill(GREEN)
        self.window_height = new_height
        self.window_width = new_width

        # Resize Game Panel
        self.game_panel_size = (int(self.window_width * GAME_WIDTH_RATIO - self.game_panel_margins[0] * 1.5),
                                                      self.window_height - self.game_panel_margins[1] * 2)
        self.game_panel = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        self.game_panel_mask = pygame.Surface(self.game_panel_size, pygame.SRCALPHA)
        pygame.draw.rect(self.game_panel_mask, (255, 255, 255, 255), (0,0)+self.game_panel_size, border_radius=PANEL_RADIUS)
        
        # Resize Control Panel
        self.manager.set_window_resolution((self.window_width, self.window_height))
        self.control_panel_pos  = (self.game_panel_size[0] + self.game_panel_margins[0] * 2, self.game_panel_margins[1])
        self.control_panel_size = (int(self.window_width * CONTROL_WIDTH_RATIO) - self.game_panel_margins[0] * 1.5,
                                                             self.window_height - self.game_panel_margins[1] * 2)
        self.control_panel.set_relative_position((self.game_panel_size[0] + self.game_panel_margins[0] * 2, self.game_panel_margins[1]))
        self.control_panel.set_dimensions(self.control_panel_size)
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

            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.stop_button:
                    self.stop_grid = not self.stop_grid
            
            self.manager.process_events(event)
            self.next_frame_time = self.slider.get_current_value()
            self.slider_label.set_text(str(self.next_frame_time))

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
