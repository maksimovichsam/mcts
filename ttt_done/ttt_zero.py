from mcts_zero import MCTSZeroNode, MCTSZero, SearchPolicy
from smaksimovich.torch_utils import BasicNN
from tictactoe import TicTacToe, TicTacToePlayer
from copy import deepcopy
import math
import torch

import torch.nn.functional as F

node_map = {}
def from_game(game: TicTacToe):
    global counter
    g = str(game)
    if g in node_map:
        return node_map[g]
    else:
        node_map[g] = TTTZeroNode(game)
        return node_map[g]

class TTTZeroNode(MCTSZeroNode):
    evaluator: BasicNN = None

    def clear_game_tree(self):
        node_map.clear()

    def __init__(self, game: TicTacToe):
        super().__init__()
        self.game = game
        self.state_tensor = None
        self.valid_actions = []

    def action_space(self):
        return self.game.board_size**2 

    def legal_actions(self) -> list[int]:
        return self.valid_actions

    def state(self):
        if self.state_tensor is None:
            self.state_tensor = torch.zeros((3, 3), dtype=torch.float)
            index_map = { None: 0, TicTacToePlayer.X: 1, TicTacToePlayer.O: 2}
            value_map = { None: 0, TicTacToePlayer.X: 1, TicTacToePlayer.O: -1}
            for j in range(self.game.board_size):
                for i in range(self.game.board_size):
                    self.state_tensor[j][i] = value_map[self.game.board[j][i]]
                    if self.game.board[j][i] is None:
                        self.valid_actions.append(j * 3 + i)
            self.state_tensor *= value_map[self.game.player]
        return self.state_tensor
                    
    def evaluate(self, state) -> tuple[list[float], float]:
        res = self.evaluator(state.reshape((-1, )))
        p, v = res[:9], res[9:]
        p = torch.softmax(p, dim=0)
        v = torch.tanh(v)
        p = p[self.valid_actions] / torch.sum(p[self.valid_actions])
        return p, v

    def children(self) -> list['MCTSZeroNode']:
        if self.game.game_over():
            return []

        children = []
        for j in range(self.game.board_size):
            for i in range(self.game.board_size):
                if self.game.board[j][i] is None:
                    g = deepcopy(self.game)
                    g.move(i, j)
                    children.append(from_game(g))

        return children

    def reward(self) -> float:
        if not self.is_terminal():
            return 0
        r = 1
        if self.game.is_tie():
            r = -1e-4
        return r
    
    def getSymmetries(self, pi):
        # mirror, rotational
        # assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = torch.zeros((9, ))
        pi_board[self.valid_actions] = pi.weights
        pi_board = pi_board.reshape((3, 3))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = torch.rot90(self.state(), i)
                newPi = torch.rot90(pi_board, i)
                if j:
                    newB = torch.fliplr(newB)
                    newPi = torch.fliplr(newPi)
                l += [(newB, SearchPolicy(newPi.ravel()))]
        return l


if __name__ == "__main__":
    import os.path
    import torch
    import random
    import numpy as np
    from smaksimovich.torch_utils import SimpleDataset

    hp = BasicNN.HyperParameters()
    hp.lr = 0.000001
    hp.iterations = 10
    hp.simulations = 600
    hp.num_episodes = 100
    hp.num_epochs = 10
    hp.batch_size = 64
    hp.buffer_size = 64 * 100000
    hp.temperature_threshold = None
    hp.c_puct = 4
    hp.weight_decay = 1e-4
    hp.lr_decay_steps = 10
    hp.gamma = 0.1
    hp.alpha = None
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    hp_string = '\n'.join(f"{k}: {v}" for k, v in hp.__dict__.items())
    print(f"Hyperparameters:\n{hp_string}")

    ttt_evaluator = BasicNN([9, 100, 100, 100, 100, 100, 10], hp)
    ttt_evaluator.load_from_file("./ttt974.pth")
    TTTZeroNode.evaluator = ttt_evaluator

    with open("tictactoe_solved.csv", "r") as file:
        x = []
        y = []
        lines = file.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            line = line.split(',')
            x_i = torch.tensor( list(map(float, line[:9])), dtype=torch.float)
            y_i = torch.tensor( list(map(float, line[9:])), dtype=torch.float)
            assert len(y_i) == 10
            x.append(x_i)
            y.append(y_i)
        action_probabilites = torch.stack([ y_i[0:9] for y_i in y ])
        max_actions = torch.max(action_probabilites, dim=1).values.reshape((-1, 1))
        max_actions = max_actions == action_probabilites
        best_actions = [[] for j in range(len(y))]
        for j in range(len(y)):
            for i in range(9):
                if max_actions[j, i]:
                    best_actions[j].append(i)

    generalization_loss = []
    def print_generalization_loss():
        with torch.no_grad():
            X = torch.stack(x).reshape((-1, 9))
            model_y = ttt_evaluator(X)
            model_probabilites = model_y[:, 0:9]
            model_actions = torch.argmax(model_probabilites, dim=1)
            assert model_actions.shape == (len(y),), f"Expected {(len(y),)} got {model_actions.shape}"

            total_correct = 0
            for j in range(len(y)):
                if model_actions[j].item() in best_actions[j]:
                    total_correct += 1

            percent_correct = total_correct / len(y)
            generalization_loss.append(percent_correct)
            print(f"total correct = {total_correct} / {len(y)} = {percent_correct}")
    print_generalization_loss()
    
    root = from_game(TicTacToe())
    losses = MCTSZero.train_evaluator(root, ttt_evaluator
        , iterations=hp.iterations
        , simulations=hp.simulations
        , num_episodes=hp.num_episodes
        , num_epochs=hp.num_epochs
        , buffer_size=hp.buffer_size
        , batch_size=hp.batch_size
        , c_puct=hp.c_puct
        , on_batch_complete=print_generalization_loss
        )
    
    from matplotlib import pyplot as plt
    plt.title("Generalization Loss")
    plt.plot(list(range(len(generalization_loss))), generalization_loss)
    plt.figure()
    plt.title("Losses")
    plt.plot(list(range(len(losses))), losses)
    plt.show()

    i = 0
    while os.path.exists(f"ttt{i}.pth"):
        i += 1

    ttt_evaluator.save_to_file(f"ttt{i}.pth")


