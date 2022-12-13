from mcts_zero import MCTSZeroNode
from smaksimovich.torch_utils import BasicNN
from tictactoe import TicTacToe, TicTacToePlayer
from snake import SnakeBoard
from copy import deepcopy
import math
import torch


class SnakeZeroNode(MCTSZeroNode):
    node_map = {}
    evaluator: BasicNN = None

    @staticmethod
    def from_game(game: SnakeBoard):
        if game not in SnakeZeroNode.node_map:
            SnakeZeroNode.node_map[game] = SnakeZeroNode(game)
        return SnakeZeroNode.node_map[game]

    def reset(self):
        super().reset()
        SnakeZeroNode.node_map = {}

    def __init__(self, game: SnakeBoard):
        super().__init__()
        self.game = game
        self.state_tensor = None

    def state(self):
        if self.state_tensor is None:
            self.state_tensor = torch.zeros((1, 4, SnakeBoard.height, SnakeBoard.width))
            self.state_tensor[1, 1, self.game.apple[1], self.game.apple[0]] = 1
            for player in self.game.players:
                head = player.tiles[0]
                self.state_tensor[1, 2, head[1], head[0]] = 1

                for i in range(len(player.tiles)):
                    self.state_tensor[1, 3, player.tiles[i][1], player.tiles[i][0]] = 1

        return self.state_tensor
                    
    def evaluate(self, state) -> tuple[list[float], float]:
        res = self.evaluator(state)
        p, v = res[:2], res[2:]
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
                    children.append(SnakeZeroNode.from_game(g))

        return children

    def reward(self) -> float:
        r = 1
        if self.game.is_tie():
            r = 0
        return r
