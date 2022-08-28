import sys
from tkinter import RIGHT
from typing import *
from collections import defaultdict, deque
import numpy as np
import random
import pygame


class Direction:
    UP    = np.array([0, -1])
    DOWN  = np.array([0, 1])
    LEFT  = np.array([-1, 0])
    RIGHT = np.array([1, 0])


class SnakePlayer:

    def __init__(self, tiles: deque, direction:Direction=Direction.LEFT):
        self.tiles = tiles # the head of the snake is at the front of the deque
        self.direction = direction
        self.dead = False

    def set_direction(self, d):
        if len(self.tiles) < 2:
            self.direction = d
        opposite_dir = self.tiles[1] - self.tiles[0]
        if not (d == opposite_dir).all():
            self.direction = d

    def get_valid_directions(self, d):
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        opposite_dir = self.tiles[1] - self.tiles[0]
        directions.remove(opposite_dir)
        assert len(directions) == 3
        return directions


    def step(self, board: 'SnakeBoard'):
        if self.dead:
            return

        new_head = self.tiles[0] + self.direction
        if board.is_in_bounds(new_head) and board.is_empty(new_head):
            self.tiles.appendleft(new_head)
            if (new_head == board.apple).all():
                board.generate_apple()
            else:
                self.tiles.pop()
        else:
            self.dead = True        

    def has_tile(self, tile):
        for t in self.tiles:
            if (t == tile).all():
                return True
        return False

    @staticmethod
    def build_snake(x: int, y: int, direction: Direction, length:int=3):
        head = np.array([x, y])
        tiles = deque()
        for i in range(length):
            tiles.append(head + direction * i)
        return SnakePlayer(tiles, direction * -1)
        

class SnakeBoard:
    width  = 20
    height = 20

    def __init__(self, players: List[SnakePlayer]):
        self.players = players
        self.generate_apple()

    def generate_apple(self):
        while True:
            apple = np.array([random.randint(0, self.width - 1), random.randint(0, self.height - 1)])
            if self.is_empty(apple):
                self.apple = apple
                break

    def step(self):
        for player in self.players:
            player.step(self)

    def is_gameover(self):
        return all(player.dead for player in self.players)

    def is_empty(self, tile):
        return not any(player.has_tile(tile) for player in self.players)

    def is_in_bounds(self, tile):
        return 0 <= tile[0] < self.width and 0 <= tile[1] < self.height


WHITE = 255, 255, 255
GREEN = 3,   138, 12
BLACK = 0,   0,   0
RED   = 255, 0,   0
BLUE  = 0,   0,   255

class SnakeViewer:
    APPLE_COLOR = GREEN
    SNAKE_COLOR = WHITE
    HEAD_COLORS = [RED, BLUE]
    GRID_COLOR  = 70, 70, 70
    LINE_WIDTH  = 1
    FONT_SIZE   = 20

    def __init__(self, board: SnakeBoard, w:int=720, h:int=720):
        pygame.init()
        self.size = w, h 
        self.font = pygame.font.SysFont("couriernew", self.FONT_SIZE)
        self.screen = pygame.display.set_mode(self.size)
        self.board = board

        self.cell_width  = w / self.board.width
        self.cell_height = h / self.board.height

    def draw(self):
        grid = [[ BLACK for i in range(self.board.width)] for j in range(self.board.height)]

        apple_x, apple_y = self.board.apple
        grid[apple_y][apple_x] = SnakeViewer.APPLE_COLOR

        for player_idx, player in enumerate(self.board.players):
            for idx, (x, y) in enumerate(player.tiles):
                if idx == 0:
                    grid[y][x] = SnakeViewer.HEAD_COLORS[player_idx]
                else:
                    grid[y][x] = SnakeViewer.SNAKE_COLOR

        for j in range(self.board.height):
            for i in range(self.board.width):
                x = i * self.cell_width
                y = j * self.cell_height
                rect = (x, y, self.cell_width, self.cell_height)
                pygame.draw.rect(self.screen, grid[j][i], rect)

        for i in range(1, self.board.width):
            p1 = (i * self.cell_width,            0)
            p2 = (i * self.cell_width, self.size[1])
            pygame.draw.line(self.screen, SnakeViewer.GRID_COLOR, p1, p2, width=SnakeViewer.LINE_WIDTH)
        
        for i in range(1, self.board.height):
            p1 = (           0, i * self.cell_height)
            p2 = (self.size[0], i * self.cell_height)
            pygame.draw.line(self.screen, SnakeViewer.GRID_COLOR, p1, p2, width=SnakeViewer.LINE_WIDTH)


        pygame.display.flip()



class HumanController:
    WASD = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]
    ARROWS = [pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT]
    keys_pressed = set()

    @staticmethod
    def check_keys():
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()
            if event.type == pygame.KEYDOWN:
                HumanController.keys_pressed.add(event.key)
            if event.type == pygame.KEYUP:
                HumanController.keys_pressed.remove(event.key)

    def __init__(self, snake, keyset=ARROWS):
        self.snake = snake
        self.key_map = defaultdict(lambda: (lambda x: 0))
        self.key_map[keyset[0]] = lambda x: x.on_up_pressed()
        self.key_map[keyset[1]] = lambda x: x.on_left_pressed()
        self.key_map[keyset[2]] = lambda x: x.on_down_pressed()
        self.key_map[keyset[3]] = lambda x: x.on_right_pressed()

    def on_right_pressed(self):
        self.snake.set_direction(Direction.RIGHT)

    def on_left_pressed(self):
        self.snake.set_direction(Direction.LEFT)

    def on_down_pressed(self):
        self.snake.set_direction(Direction.DOWN)
    
    def on_up_pressed(self):
        self.snake.set_direction(Direction.UP)

    def make_move(self):
        HumanController.check_keys()
        for key in HumanController.keys_pressed:
            self.key_map[key](self)

if __name__ == "__main__":
    import time

    sleep_time = 0.075
    players = [SnakePlayer.build_snake(8, 8, Direction.LEFT),
               SnakePlayer.build_snake(12, 12, Direction.RIGHT)]
    controllers = [
        HumanController(players[0], keyset=HumanController.ARROWS),
        HumanController(players[1], keyset=HumanController.WASD)
    ]
    board = SnakeBoard(players)
    viewer = SnakeViewer(board)

    while not board.is_gameover():
        viewer.draw()

        for i in range(10):
            for controller in controllers:
                controller.make_move()
            time.sleep(sleep_time / 10)

        board.step()

