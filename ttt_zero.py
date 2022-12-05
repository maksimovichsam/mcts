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
        TTTZeroNode.node_map = {}

    def __init__(self, game: TicTacToe):
        super().__init__()
        self.game = game
        self.state_tensor = None
        self.valid_actions = []

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
                    children.append(TTTZeroNode.from_game(g))

        return children

    def reward(self) -> float:
        r = 1
        if self.game.is_tie():
            r = 0
        return r


if __name__ == "__main__":
    import os.path

    hp = BasicNN.HyperParameters()
    hp.lr = 0.01
    ttt_evaluator = BasicNN([27, 100, 100, 100, 100, 10], hp)
    TTTZeroNode.evaluator = ttt_evaluator

    # ttt_evaluator.load_from_file("ttt3.pth")
    root = TTTZeroNode.from_game(TicTacToe())
    MCTSZero.train_evaluator(root, ttt_evaluator, iterations=400, simulations=100)

    i = 0
    while os.path.exists(f"ttt{i}.pth"):
        i += 1

    ttt_evaluator.save_to_file(f"ttt{i}.pth")


