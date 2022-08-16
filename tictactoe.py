from typing import *
from enum import Enum
from abc import ABC, abstractmethod, abstractstaticmethod

WHITE = 255, 255, 255
GREEN = 3, 138, 12
BLACK = 0, 0, 0
RED = 255, 0, 0

class TicTacToePlayer(Enum):
    X = 0
    O = 1


class TicTacToeWinPattern:
    def __init__(self, win_size=3):
        self.win_size = win_size

    @abstractmethod
    def pattern(self, i, j):
        pass

class Horizontal(TicTacToeWinPattern):
    def pattern(self, i, j):
        return iter((i + k, j) for k in range(1, self.win_size))

class Vertical(TicTacToeWinPattern):
    def pattern(self, i, j):
        return iter((i, j + k) for k in range(1, self.win_size))

class DiagonalLeftDown(TicTacToeWinPattern):
    def pattern(self, i, j):
        return iter((i + k, j + k) for k in range(1, self.win_size))

class DiagonalLeftUp(TicTacToeWinPattern):
    def pattern(self, i, j):
        return iter((i + k, j - k) for k in range(1, self.win_size))


class TicTacToe:
    board_size: int = 3
    win_size: int = 3
    patterns: List[TicTacToeWinPattern] = [
        Horizontal(win_size),
        Vertical(win_size),
        DiagonalLeftDown(win_size),
        DiagonalLeftUp(win_size)
    ]

    def __init__(self):
        self.board = [[None for i in range(self.board_size)] for j in range(self.board_size)]
        self.spaces_left = self.board_size**2
        self.player = TicTacToePlayer.X

    def __hash__(self):
        return hash(tuple(item for row in self.board for item in row))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def move(self, x:int, y:int):
        if self.get(x, y) is not None:
            return
        self.set(x, y, self.player)
        self.player = TicTacToePlayer((self.player.value + 1) % 2)
        self.spaces_left -= 1

    def in_bounds(self, x:int, y:int):
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def get(self, x:int, y:int):
        return self.board[y][x]

    def game_over(self):
        return self.spaces_left == 0 or self.winner() is not None

    def winner(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                player = self.get(i, j)

                for pattern in self.patterns:
                    pattern_has_winner = True
                    for (x_i, y_i) in pattern.pattern(i, j):
                        if not self.in_bounds(x_i, y_i) or self.get(x_i, y_i) != player:
                            pattern_has_winner = False
                            break
                    
                    if pattern_has_winner:
                        return player
        
        return None

    def set(self, x:int, y:int, player: TicTacToePlayer):
        self.board[y][x] = player

    def __str__(self):
        s = ""
        item_map = {TicTacToePlayer.X: "X", TicTacToePlayer.O: "O", None: " "}
        for idx, row in enumerate(self.board):
            s += "|".join([item_map[item] for item in row])
            if idx + 1 < len(self.board):
                s += "\n" + ("-" * (2 * self.board_size - 1)) + "\n"
        return s

import pygame
import math

class TicTacToeViewer:
    FONT_SIZE = 24
    BACKGROUND_COLOR = GREEN
    LINE_WIDTH = 20

    def __init__(self, game, w=720, h=720):
        pygame.init()
        self.size = w, h 
        self.font = pygame.font.SysFont("couriernew", self.FONT_SIZE)
        self.screen = pygame.display.set_mode(self.size)
        self.game = game

        self.cell_width  = w / self.game.board_size
        self.cell_height = h / self.game.board_size

    def get_board_coords(self, x, y):
        i = math.floor((x / self.size[0]) * self.game.board_size)
        j = math.floor((y / self.size[1]) * self.game.board_size)
        return (i, j)

    def draw(self):
        self.screen.fill(self.BACKGROUND_COLOR)

        for i in range(1, self.game.board_size):
            x = (i / self.game.board_size) * self.size[0]
            y = (i / self.game.board_size) * self.size[1]
            pygame.draw.line(self.screen, WHITE, (x, 0), (x, self.size[1]), width=self.LINE_WIDTH)
            pygame.draw.line(self.screen, WHITE, (0, y), (self.size[0], y), width=self.LINE_WIDTH)

        for j in range(self.game.board_size):
            for i in range(self.game.board_size):
                tile = self.game.get(i, j)
                if tile is None:
                    continue
                
                x = (i / self.game.board_size) * self.size[0]
                y = (j / self.game.board_size) * self.size[1]
                if tile == TicTacToePlayer.X:
                    self.draw_x(x, y)
                elif tile == TicTacToePlayer.O:
                    self.draw_o(x, y)
                else:
                    raise Exception(f"draw(): Unknown player {tile}")

        pygame.display.flip()

    def draw_x(self, x, y):
        p1 = (x, y)
        p2 = (x + self.cell_width, y + self.cell_height)
        pygame.draw.line(self.screen, WHITE, p1, p2, width=self.LINE_WIDTH)
        p1 = (x, y + self.cell_height)
        p2 = (x + self.cell_width, y)
        pygame.draw.line(self.screen, WHITE, p1, p2, width=self.LINE_WIDTH)

    def draw_o(self, x, y):
        p1 = (x, y, self.cell_width, self.cell_height)
        pygame.draw.arc(self.screen, WHITE, p1, 0, math.pi * 2, width=self.LINE_WIDTH)


if __name__ == "__main__":
    screen_size = 720

    game = TicTacToe()
    viewer = TicTacToeViewer(game, w=screen_size, h=screen_size)

    from ttt_mcts import TicTacToeNode
    save_file = "ttt.pickle"
    TicTacToeNode.load(save_file)

    computer = TicTacToePlayer.X
    human = TicTacToePlayer.O

    run = True
    def make_move(i, j):
        global run
        game.move(i, j)
        if game.game_over():
            winner = game.winner()
            if winner is not None:
                print(f"Player {winner.name} won tic tac toe!")
            else:
                print(f"Tic-tac-toe ended in a tie")
            run = False

    while run:
        if game.player == computer:
            i, j = TicTacToeNode.pick_move(game)
            make_move(i, j)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.player != human:
                    continue
                x_m, y_m = pygame.mouse.get_pos()
                i, j = viewer.get_board_coords(x_m, y_m)
                make_move(i, j)

        viewer.draw()
    
    print("Exiting")