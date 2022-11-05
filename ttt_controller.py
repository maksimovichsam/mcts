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


from ttt_mcts import TicTacToeNode

class MCTSPlayer(TTTController):

    def __init__(self):
        save_file = "ttt.pickle"
        TicTacToeNode.load(save_file)

    def pick_move(self, game: TicTacToe):
        return TicTacToeNode.pick_move(game)

import torch
import random
import math
from smaksimovich.torch_utils import BasicNN

class MCTSZeroPlayer:

    def __init__(self, save_file):
        self.ttt_evaluator = BasicNN([27, 100, 100, 100, 100, 10])
        self.ttt_evaluator.load_from_file(save_file)

    def pick_move(self, game: TicTacToe):
        state = torch.zeros((3, 3, 3), dtype=torch.float)
        index_map = { None: 0, TicTacToePlayer.X: 1, TicTacToePlayer.O: 2}
        valid_actions = []
        for j in range(game.board_size):
            for i in range(game.board_size):
                state[j][i][index_map[game.board[j][i]]] = 1
                if game.board[j][i] is None:
                    valid_actions.append(j * 3 + i)
        state = state.reshape((-1, ))
        res = self.ttt_evaluator(state)
        p, v = res[:9], res[9:]
        v = torch.arctan(v) / (math.pi / 2)
        p[valid_actions] = torch.softmax(p[valid_actions], dim=0)
        board_p = [[p[j * 3 + i] if (j * 3 + i) in valid_actions else 0 for i in range(game.board_size)] for j in range(game.board_size)]
        print(f"TTTZero value: {v.item():5>.4f}")
        print('\n'.join([' '.join([f"{x:>3.2f}" for x in row]) for row in board_p]))
        
        action = random.choices(population=valid_actions, weights=p[valid_actions], k=1)[0]
        print(f"Chose action {action} = {action % 3, action // 3}")
        return action % 3, action // 3 # x, y



if __name__ == "__main__":
    screen_size = 720

    game = TicTacToe()
    viewer = TicTacToeViewer(game, w=screen_size, h=screen_size)

    human_controller = HumanController(viewer)
    computer_controller = MCTSZeroPlayer("./ttt2.pth")

    player_map = {
        TicTacToePlayer.X:
            # human_controller
            computer_controller
        ,
        TicTacToePlayer.O:
            human_controller
            # computer_controller
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