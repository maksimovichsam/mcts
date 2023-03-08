from tictactoe import TicTacToe, TicTacToePlayer, TicTacToeViewer
from abc import ABC, abstractmethod

class TTTController(ABC):

    @abstractmethod
    def pick_move(self, game: TicTacToe):
        pass

import pygame

class HumanController(TTTController):
    
    def __init__(self, viewer):
        self.viewer = viewer

    def pick_move(self, game: TicTacToe):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x_m, y_m = pygame.mouse.get_pos()
                    return self.viewer.get_board_coords(x_m, y_m)


import torch
import random
import math
from smaksimovich.torch_utils import BasicNN
import ttt_zero
import mcts_zero

class MCTSZeroPlayer:

    def __init__(self, save_file):
        self.ttt_evaluator = BasicNN([9, 100, 100, 100, 100, 100, 10])
        self.ttt_evaluator.load_from_file(save_file)
        ttt_zero.TTTZeroNode.evaluator = self.ttt_evaluator

    def pick_move(self, game: TicTacToe):
        node = ttt_zero.from_game(game)
        p = mcts_zero.MCTSZero.search(node, simulations=800).weights
        print("simulation values")
        board_p = torch.zeros((game.board_size**2, ))
        board_p[node.valid_actions] = p
        board_p = board_p.reshape((3, 3))
        print('\n'.join([' '.join([f"{x:>3.2f}" for x in row]) for row in board_p]))

        p, v = node.evaluate(node.state())
        print(f"TTTZero value: {v.item():5>.4f}")
        print(str(node.game))
        print(','.join(list(map(str, node.state().reshape((-1,)).tolist()))))
        board_p = torch.zeros((game.board_size**2, ))
        board_p[node.valid_actions] = p
        board_p = board_p.reshape((3, 3))
        print('\n'.join([' '.join([f"{x:>3.2f}" for x in row]) for row in board_p]))
        
        action = node.valid_actions[torch.argmax(p).item()]
        print(f"Chose action with highest p {action} = {action % 3, action // 3}")
        return action % 3, action // 3 # x, y



if __name__ == "__main__":
    screen_size = 720

    game = TicTacToe()
    viewer = TicTacToeViewer(game, w=screen_size, h=screen_size)

    human_controller = HumanController(viewer)
    computer_controller = MCTSZeroPlayer("./ttt974.pth")

    player_map = {
        TicTacToePlayer.X:
            human_controller
            # computer_controller
        ,
        TicTacToePlayer.O:
            # human_controller
            computer_controller
    }

    run = True
    viewer.draw()
    while run:
        i, j = player_map[game.player].pick_move(game)

        game.move(i, j)
        if game.game_over():
            winner = game.winner()
            if winner is not None:
                print(f"Player {winner.name} won tic tac toe!")
            else:
                print(f"Tic-tac-toe ended in a tie")
            run = False

        viewer.draw()
    
    print(game)
    print("Exiting")