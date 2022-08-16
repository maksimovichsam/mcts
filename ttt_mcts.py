from collections import defaultdict
from mcts import MCTS, MCTSNode
from tictactoe import TicTacToe, TicTacToePlayer
from copy import deepcopy
import pickle

class TicTacToeNode(MCTSNode):
    node_map = {}

    @staticmethod
    def save(file_name):
        with open(file_name, "wb") as file:
            pickle.dump(TicTacToeNode.node_map, file)

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as file:
            TicTacToeNode.node_map = pickle.load(file)

    @staticmethod
    def from_game(game: TicTacToe):
        if game not in TicTacToeNode.node_map:
            TicTacToeNode.node_map[game] = TicTacToeNode(game)
        return TicTacToeNode.node_map[game]

    def __init__(self, game):
        super().__init__()
        self.game = game

    def children(self):
        if self.game.game_over():
            return []

        children = []
        for j in range(self.game.board_size):
            for i in range(self.game.board_size):
                if self.game.board[j][i] is None:
                    g = deepcopy(self.game)
                    g.move(i, j)
                    children.append(TicTacToeNode.from_game(g))

        return children

    def reward(self):
        winner = self.game.winner()
        if winner is None:
            return 0
        return 1 if winner == TicTacToePlayer.X else -1

    @staticmethod
    def pick_move(game):
        node = TicTacToeNode.from_game(game)
        value_array = [[None for i in range(node.game.board_size)] for j in range(node.game.board_size)]

        children = node.node_children()
        idx = 0
        best_idx = None
        best_value = None
        for j in range(node.game.board_size):
            for i in range(node.game.board_size):
                if node.game.board[j][i] is not None:
                    value_array[j][i] = node.game.board[j][i]
                    continue

                if children[idx].visits == 0:
                    idx += 1
                    continue

                value = children[idx].total_reward / children[idx].visits
                if best_value is None or value > best_value:
                    best_value = value
                    best_idx = (i, j)

                value_array[j][i] = value
                idx += 1

        for row in value_array:
            print(row)

        return best_idx

if __name__ == "__main__":
    import pickle
    
    save_file = "ttt.pickle"
    root = TicTacToeNode.from_game(TicTacToe())

    playouts = 10_000
    for i in range(playouts):
        MCTS.select(root, num_rollouts=1000)

        if i % (playouts // 200) == 0:
            print(f"Game {i + 1:>8}, {i / playouts * 100:>5.1f}% complete")

    print(TicTacToeNode.pick_move(root.game))
    TicTacToeNode.save(save_file)

