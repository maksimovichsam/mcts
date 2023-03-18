
def black_box_function(c_puct, simulations, episodes, epochs):
    import torch
    import random
    import numpy as np
    from mcts_zero import MCTSZero
    from smaksimovich.torch_utils import BasicNN
    from snake_zero import SnakeNet, SnakeZeroNode

    hp = BasicNN.HyperParameters()
    hp.lr = 0.001
    hp.iterations = 5
    hp.simulations = int(simulations)
    hp.num_episodes = int(episodes)
    hp.num_epochs = int(epochs)
    hp.batch_size = 64
    hp.buffer_size = 64 * 1000
    hp.temperature_threshold = None
    hp.c_puct = c_puct
    hp.weight_decay = 1e-4
    hp.lr_decay_steps = 10000
    hp.gamma = 0.1
    hp.alpha = None
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    hp_string = '\n'.join(f"{k}: {v}" for k, v in hp.__dict__.items())
    print(f"Hyperparameters:\n{hp_string}")

    snake_net = SnakeNet()
    snake_net.hp = hp
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

    rewards = rewards[-50:]
    return sum(rewards) / len(rewards)

from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {
    'c_puct': (0, 15),
    'simulations': (5, 20),
    'episodes': (5, 50),
    'epochs': (5, 40),
    }

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