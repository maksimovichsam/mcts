from mcts_zero import MCTSZeroNode, MCTSZero
from smaksimovich.torch_utils import BasicNN
from snake import SnakeBoard, Direction, SnakePlayer, DIRECTIONS
from copy import deepcopy
import math
import torch
from torch.nn import functional as F
from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self, in_f, out_f, non_linear):
        super(ResidualBlock, self).__init__()
        self.non_linear = non_linear()
        self.conv1 = nn.Conv2d(in_f, out_f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_f, out_f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_f, out_f, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.non_linear(self.conv1(x))
        x = x + self.non_linear(self.conv2(x))
        return x + self.conv3(x)

class SnakeNet(nn.Module):

    def __init__(self):
        super(SnakeNet, self).__init__()
        self.non_linear = nn.ReLU

       # same padding => 2*p + k + 1 = 0 => p = (k - 1) / 2
        # (32, 32, 2, 2) -> (16, 16, 2, 2) -> (8, 8, 2, 2) -> (4, 4, 2, 2)
        self.cnn = nn.Sequential(
            # 8, 8
            # ResidualBlock(64, 128, self.non_linear),
            nn.Conv2d(4, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4, 4
            nn.BatchNorm2d(1024),
            # ResidualBlock(32, 64, self.non_linear),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2, 2
        )
        self.cnn_out = self.cnn(torch.randn((1, 4, 8, 8))).numel()
        assert self.cnn_out == 2*2*2048, f"Got cnn_out={self.cnn_out}"
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_out, 2048),
            self.non_linear(),
            nn.Linear(2048, 512),
            self.non_linear(),
            nn.Linear(512, 64),
            self.non_linear(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x.view((-1, self.cnn_out)))
        return x
    
    def save_to_file(self, file_str):
        torch.save(self.state_dict(), file_str)

    def load_from_file(self, file_str):
        self.load_state_dict(torch.load(file_str))


class SnakeNetController:

    def __init__(self, snake, load_file=None):
        self.model = SnakeNet()
        if load_file is not None:
            self.model.load_from_file(load_file)
        SnakeZeroNode.evaluator = self.model        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"SnakeNet model has {num_params:,} parameters")
        self.snake = snake

    def make_move(self, board):
        node = SnakeZeroNode.from_game(board)
        node.is_terminal()
        p = MCTSZero.search(node, None, gamma=0.997, simulations=5, temperature=1)
        pi = torch.zeros((node.action_space(), ))
        pi[node.legal_actions()] = p.weights
        action = torch.argmax(pi)
        self.snake.set_direction(DIRECTIONS[action])


class SnakeZeroNode(MCTSZeroNode):
    node_map = {}
    evaluator: SnakeNet

    @staticmethod
    def init_game():
        players = [SnakePlayer.build_snake(2, 2, Direction.RIGHT)]
        return SnakeZeroNode.from_game(SnakeBoard(players))

    @staticmethod
    def from_game(game: SnakeBoard):
        s = str(game)
        exists = s in SnakeZeroNode.node_map
        if not exists:
            SnakeZeroNode.node_map[s] = SnakeZeroNode(game)
        return SnakeZeroNode.node_map[s]

    def clear_game_tree(self):
        SnakeZeroNode.node_map = {}
        return SnakeZeroNode.init_game()

    def action_space(self) -> int:
        return 4

    def __init__(self, game: SnakeBoard):
        super().__init__()
        self.game = game
        self.state_tensor = None

    def state(self, device):
        if self.state_tensor is None:
            INDICES = [0, 1, 2, 3]
            BLANK, APPLE, HEAD, SNAKE = INDICES
            self.state_tensor = torch.zeros((1, len(INDICES), SnakeBoard.height, SnakeBoard.width), dtype=torch.float, device=device)
            self.state_tensor[:, BLANK, :, :] = 1
            self.state_tensor[:, BLANK, self.game.apple[1], self.game.apple[0]] = 0
            self.state_tensor[:, APPLE, self.game.apple[1], self.game.apple[0]] = 1
            for player in self.game.players:
                head = player.tiles[0]
                self.state_tensor[:, BLANK, head[1], head[0]] = 0
                self.state_tensor[:, HEAD, head[1], head[0]] = 1

                for i in range(1, len(player.tiles)):
                    self.state_tensor[:, BLANK , player.tiles[i][1], player.tiles[i][0]] = 0
                    self.state_tensor[:, SNAKE , player.tiles[i][1], player.tiles[i][0]] = 1
            self.valid_actions = self.game.legal_actions()

        return self.state_tensor
    
    def legal_actions(self):
        return self.game.legal_actions()
                    
    def evaluate(self, state) -> tuple[list[float], float]:
        res = SnakeZeroNode.evaluator(state)
        p, v = res[0, :4], res[0, 4:]
        p = torch.softmax(p, dim=0)
        v = torch.relu(v + 1) - 1
        # v = torch.tanh(v)
        # v = torch.sigmoid(v)
        # assert 0 <= v.item() <= 1
        p = p[self.valid_actions] / torch.sum(p[self.valid_actions])
        return p, v

    def children(self) -> list['MCTSZeroNode']:
        if self.game.is_gameover():
            return []

        children = []
        actions = self.game.legal_actions()
        original_snake_len = len(self.game.players[0].tiles)
        for action in actions:
            new_game = deepcopy(self.game)
            new_game.players[0].set_direction(DIRECTIONS[action], assert_valid=True)
            new_game.step()
            snake_grew = len(new_game.players[0].tiles) > original_snake_len
            snake_won = len(new_game.players[0].tiles) >= new_game.width * new_game.height
            reward = 10 if snake_won else -1 if new_game.is_gameover() else 1 if snake_grew else -0.01
    
            children.append((reward, SnakeZeroNode.from_game(new_game)))

        return children

    def reward(self) -> float:
        assert False


def main():
    import os.path
    import torch
    import random
    import numpy as np
    from mcts_zero import MCTSZero

    hp = BasicNN.HyperParameters()
    hp.cuda = True
    hp.lr = 0.001
    hp.iterations = 40
    hp.simulations = 10
    hp.num_episodes = 25
    hp.num_epochs = 15
    hp.batch_size = 64
    hp.buffer_size = 64 * 1000
    hp.temperature_threshold = None
    hp.c_puct = 4
    hp.weight_decay = 1e-4
    hp.lr_decay_steps = 10000
    hp.gamma = 0.1
    hp.alpha = None
    hp.checkpoint = 5
    hp.save_file = "./replay_buffer.pickle"
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    hp_string = '\n'.join(f"{k}: {v}" for k, v in hp.__dict__.items())
    print(f"Hyperparameters:\n{hp_string}")

    snake_net = SnakeNet()
    # snake_net.load_from_file('./snake0.pth')
    snake_net.hp = hp
    if hp.cuda:
        assert torch.cuda.is_available()
        snake_net.to('cuda')
    SnakeZeroNode.evaluator = snake_net

    root = SnakeZeroNode.init_game()
    losses, rewards = MCTSZero.train_evaluator(root, snake_net
        , iterations=hp.iterations
        , simulations=hp.simulations
        , num_episodes=hp.num_episodes
        , num_epochs=hp.num_epochs
        , buffer_size=hp.buffer_size
        , batch_size=hp.batch_size
        , c_puct=hp.c_puct
        )
    
    from matplotlib import pyplot as plt
    window_size = 50
    avg_rewards = []
    for i in range(len(rewards)):
        start = max(0, i - window_size)
        end = i + 1
        avg = sum(rewards[start:end]) / (end - start) 
        avg_rewards.append(avg)
    plt.title("Rewards")
    plt.plot(list(range(len(rewards))), rewards, 'g.')
    plt.plot(list(range(len(rewards))), avg_rewards)
    plt.figure()
    plt.title("Losses")
    plt.plot(list(range(len(losses))), losses, 'b.')
    plt.show()

    i = 0
    while os.path.exists(f"snake{i}.pth"):
        i += 1

    snake_net.save_to_file(f"snake{i}.pth")


if __name__ == "__main__":
    main()