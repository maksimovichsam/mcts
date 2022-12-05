from collections import defaultdict
from mcts import MCTS, MCTSNode
from snake import SnakeBoard
from copy import deepcopy
import pickle

class SnakeNode(MCTSNode):
    node_map = {}

    @staticmethod
    def save(file_name):
        with open(file_name, "wb") as file:
            pickle.dump(SnakeNode.node_map, file)

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as file:
            SnakeNode.node_map = pickle.load(file)

    @staticmethod
    def from_game(game: SnakeBoard):
        if game not in SnakeNode.node_map:
            SnakeNode.node_map[game] = SnakeNode(game)
        return SnakeNode.node_map[game]

    def __init__(self, game: SnakeBoard):
        super().__init__()
        self.game = game

    def children(self):
        pass

    def reward(self):
        pass

    @staticmethod
    def pick_move(game):
        pass