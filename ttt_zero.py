from mcts_zero import MCTSZeroNode, MCTSZero
from smaksimovich.torch_utils import BasicNN
from tictactoe import TicTacToe, TicTacToePlayer
from copy import deepcopy
import math
import torch


class TTTZeroNode(MCTSZeroNode):
    node_map = {}
    evaluator: BasicNN = None

    @staticmethod
    def from_game(game: TicTacToe):
        if game not in TTTZeroNode.node_map:
            TTTZeroNode.node_map[game] = TTTZeroNode(game)
        return TTTZeroNode.node_map[game]

    def reset(self):
        super().reset()
        TTTZeroNode.node_map.clear()

    def __init__(self, game: TicTacToe):
        super().__init__()
        self.game = game
        self.state_tensor = None
        self.valid_actions = []

    def action_space(self):
        return self.game.board_size**2 

    def state(self):
        if self.state_tensor is None:
            self.state_tensor = torch.zeros((3, 3, 3), dtype=torch.float)
            index_map = { None: 0, TicTacToePlayer.X: 1, TicTacToePlayer.O: 2}
            for j in range(self.game.board_size):
                for i in range(self.game.board_size):
                    self.state_tensor[j][i][index_map[self.game.board[j][i]]] = 1
                    if self.game.board[j][i] is None:
                        self.valid_actions.append(j * 3 + i)
            self.state_tensor = self.state_tensor.reshape((-1, ))
        return self.state_tensor
                    
    def evaluate(self, state) -> tuple[list[float], float]:
        res = self.evaluator(state)
        p, v = res[:9][self.valid_actions], res[9:]
        p = torch.softmax(p, dim=0)
        v = torch.arctan(v) / (math.pi / 2)
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
                    # children.append(TTTZeroNode.from_game(g))
                    children.append((j * 3 + i, TTTZeroNode.from_game(g)))

        return children

    def reward(self) -> float:
        r = 1
        if self.game.is_tie():
            r = 0
        return r


if __name__ == "__main__":
    import os.path
    import torch
    import random

    hp = BasicNN.HyperParameters()
    hp.lr = 0.0001
    hp.iterations = 50
    hp.simulations = 1000
    hp.num_episodes = 20
    hp.num_epochs = 10
    hp.buffer_size = 64 * 10
    hp.batch_size = 64
    hp.temperature = 1
    hp.c_puct = 4
    hp.stop_loss = 0.001
    hp.weight_decay = 10e-4
    torch.manual_seed(0)
    random.seed(0)
    hp_string = '\n'.join(f"{k}: {v}" for k, v in hp.__dict__.items())
    print(f"Hyperparameters:\n{hp_string}")

    ttt_evaluator = BasicNN([27, 100, 100, 100, 100, 9], hp)
    TTTZeroNode.evaluator = ttt_evaluator

    # ttt_evaluator.load_from_file("ttt9.pth")
    root = TTTZeroNode.from_game(TicTacToe())
    MCTSZero.train_evaluator(root, ttt_evaluator
        , iterations=hp.iterations
        , simulations=hp.simulations
        , num_episodes=hp.num_episodes
        , num_epochs=hp.num_epochs
        , buffer_size=hp.buffer_size
        , batch_size=hp.batch_size
        , temperature=hp.temperature
        , c_puct=hp.c_puct
        , stop_loss=hp.stop_loss
        )

    i = 0
    while os.path.exists(f"ttt{i}.pth"):
        i += 1

    ttt_evaluator.save_to_file(f"ttt{i}.pth")


