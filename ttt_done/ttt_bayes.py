import os.path
import torch
import random
import numpy as np
from smaksimovich.torch_utils import SimpleDataset, BasicNN
from ttt_zero import TTTZeroNode
from mcts_zero import MCTSZero
from tictactoe import TicTacToe
import ttt_zero


with open("tictactoe_solved.csv", "r") as file:
    x = []
    y = []
    lines = file.read().split('\n')
    for line in lines:
        if len(line) == 0:
            continue
        line = line.split(',')
        x_i = torch.tensor( list(map(float, line[:27])), dtype=torch.float)
        y_i = torch.tensor( list(map(float, line[27:])), dtype=torch.float)
        assert len(y_i) == 10
        new_x_i = torch.zeros((9, ))
        for i in range(0, 27, 3):
            if x_i[i + 1] == 1:
                new_x_i[i // 3] = 1
            elif x_i[i + 2] == 1:
                new_x_i[i // 3] = -1
        new_x_i = new_x_i.reshape((3, 3))
        x.append(new_x_i)
        y.append(y_i)
    dataset = SimpleDataset(x, y)
    action_probabilites = torch.stack([ y_i[0:9] for y_i in y ])
    max_actions = torch.max(action_probabilites, dim=1).values.reshape((-1, 1))
    max_actions = max_actions == action_probabilites
    best_actions = [[] for j in range(len(y))]
    for j in range(len(y)):
        for i in range(9):
            if max_actions[j, i]:
                best_actions[j].append(i)

def generalization_loss(model):
    with torch.no_grad():
        X = torch.stack(x).reshape((-1, 9))
        model_y = model(X)
        model_probabilites = model_y[:, 0:9]
        model_actions = torch.argmax(model_probabilites, dim=1)
        assert model_actions.shape == (len(y),), f"Expected {(len(y),)} got {model_actions.shape}"

        total_correct = 0
        for j in range(len(y)):
            if model_actions[j].item() in best_actions[j]:
                total_correct += 1

        percent_correct = total_correct / len(y)
        return percent_correct
    

def black_box_function(c_puct):
    hp = BasicNN.HyperParameters()
    hp.lr = 0.001
    hp.iterations = 10
    hp.simulations = 25
    hp.num_episodes = 25
    hp.num_epochs = 10
    hp.batch_size = 64
    hp.buffer_size = 2000000
    hp.temperature = 1
    hp.c_puct = c_puct
    hp.weight_decay = 0
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    ttt_evaluator = BasicNN([9, 100, 100, 100, 100, 100, 11], hp)
    TTTZeroNode.evaluator = ttt_evaluator

    root = ttt_zero.from_game(TicTacToe())
    MCTSZero.train_evaluator(root, ttt_evaluator
        , iterations=hp.iterations
        , simulations=hp.simulations
        , num_episodes=hp.num_episodes
        , num_epochs=hp.num_epochs
        , buffer_size=hp.buffer_size
        , batch_size=hp.batch_size
        , temperature=hp.temperature
        , c_puct=hp.c_puct
        )
    
    ttt_evaluator.save_to_file(f"tttcpuct{c_puct:.2f}.pth")

    return generalization_loss(ttt_evaluator)

from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'c_puct': (0, 10)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=0,
)

optimizer.maximize(
    init_points=10,
    n_iter=10,
)

print(optimizer.max)