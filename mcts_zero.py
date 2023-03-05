from abc import ABC, abstractmethod
from smaksimovich import Timer
from torch.distributions import Dirichlet
import torch.nn.functional as F
from collections import deque
import random
import math
import torch
import torch.nn as nn
import numpy as np


class SearchPolicy:
    
    def __init__(self, weights: list[float]):
        self.weights = weights
        self.indices = list(range(len(self.weights)))

    def pick(self) -> int:
        return random.choices(population=self.indices, weights=self.weights, k=1)[0]


class MCTSZeroNode(ABC):
    c_puct = 2
    noise_scale = 0.10
    
    def __init__(self):
        self.reset()

    def node_children(self) -> list['MCTSZeroNode']:
        if self.nodes is not None:
            return self.nodes
        self.nodes = self.children()
        return self.nodes

    def is_terminal(self) -> bool:
        return len(self.node_children()) == 0

    def is_leaf(self) -> bool:
        return self.visits is None

    def select(self, noise=3, epsilon=0.25) -> tuple[int, 'MCTSZeroNode']:
        # noise = Dirichlet(noise * torch.ones((len(self.visits), ))).sample()
        # selection = self.q + MCTSZeroNode.c_puct * ((1 - epsilon) * self.p + epsilon * noise) * math.sqrt(self.total_visits + 0.01) / (1 + self.visits)
        selection = self.q + MCTSZeroNode.c_puct * self.p * math.sqrt(self.total_visits + 1e-8) / (1 + self.visits)
        action = torch.argmax(selection, dim=0)
        return action, self.node_children()[action]

    def expand(self) -> float:
        self.p, self.v = self.evaluate(self.state())
        self.visits      = torch.zeros((len(self.p), ), dtype=torch.float)
        self.total_value = torch.zeros((len(self.p), ), dtype=torch.float)
        self.q           = torch.zeros((len(self.p), ), dtype=torch.float)
        return self.v

    def backup(self, child_index: int, value: float) -> None:
        self.total_value[child_index:child_index+1] += value
        self.visits[child_index:child_index+1] += 1
        self.total_visits += 1
        self.q[child_index:child_index+1] = self.total_value[child_index] / self.visits[child_index]

    def search_policy(self, temperature: float=1) -> SearchPolicy:
        if temperature == 0:
            weights = torch.zeros_like(self.visits)
            weights[torch.argmax(self.visits, dim=0)] = 1
        else:
            weights = torch.pow(self.visits, 1 / temperature)
            weights /= torch.sum(weights)
        return SearchPolicy(weights)

    def reset(self):
        """ A deletes the search tree and resets nodes statistics"""
        self.visits = None
        self.total_visits = 0
        self.total_value = None
        self.p = None
        self.q = None
        self.v = None
        self.nodes = None

    @abstractmethod
    def state(self):
        """ Returns the tensor representation of this node
        """
        pass

    @abstractmethod
    def evaluate(self, state) -> tuple[list[float], float]:
        """ Computes tensors p, v for the given state
            - p = normalized probabilities of legal moves
            - v = value of the state
        """
        pass

    @abstractmethod
    def children(self) -> list['MCTSZeroNode']:
        """ Returns the sub states after taking one action from 
            the given state
        """
        pass

    @abstractmethod
    def reward(self) -> float:
        """ Returns the game reward from the perspective of the player
            which made the last move
        """
        pass



class MCTSZero:

    @staticmethod
    def search(node: MCTSZeroNode, simulations:int=100, temperature:float=1) -> SearchPolicy:
        root: MCTSZeroNode = node
        for i in range(simulations):
            path: list[tuple[int, MCTSZeroNode]] = []
            while not node.is_leaf() and not node.is_terminal():
                action, new_node = node.select()
                path.append((action, node))
                node = new_node
           
            v = node.reward() if node.is_terminal() else -node.expand()
            for index, (action, parent) in enumerate(reversed(path)):
                parent.backup(action, v * (-1)**(index % 2))
            
            node = root

        return root.search_policy(temperature=temperature)

    @staticmethod
    def play(node: MCTSZeroNode, **kwargs) -> tuple[list[MCTSZeroNode], list[SearchPolicy]]:
        while True:
            examples = []
            actions = []
            while not node.is_terminal():
                policy = MCTSZero.search(node, **kwargs)
                examples.append([node, node.state(), policy, None])

                # sym = node.getSymmetries(policy)
                # for b, p in sym:
                #     examples.append([node, b, p, None])

                action = policy.pick()
                actions.append(action)
                node = node.node_children()[action]
            r = node.reward()
            for idx, example in enumerate(reversed(examples)):
                assert (-1)**(example[0].game.player == node.game.player) == (-1)**(idx % 2)
                example[3] = (-1)**(example[0].game.player == node.game.player) * r
            return examples
        
    @staticmethod
    def train_evaluator(root: MCTSZeroNode, evaluator, 
            num_epochs=5, num_episodes=1, iterations=10, batch_size=256, 
            buffer_size=512, c_puct=4, stop_loss=0, on_batch_complete=lambda:0, **kwargs):
        MCTSZeroNode.c_puct = c_puct
        optimizer = torch.optim.AdamW(evaluator.parameters(), lr=evaluator.hp.lr,
            weight_decay=evaluator.hp.weight_decay)

        game_buffer = deque(maxlen=buffer_size)

        for iteration in range(iterations):
            Timer.start("iteration")

            for episodes in range(num_episodes):
                examples = MCTSZero.play(root, **kwargs)
                game_buffer.extend(examples)
                root.reset()
                root.clear_game_tree()

            game_buffer_order = list(range(len(game_buffer)))

            num_batches = math.ceil(len(game_buffer) / batch_size)
            evaluator.train()

            for epoch in range(num_epochs):
                random.shuffle(game_buffer_order)

                for batch_number in range(num_batches):
                    loss = torch.tensor([0.0])
                    losses = []

                    batch_start = batch_number * batch_size
                    batch_end = min(batch_start + batch_size, len(game_buffer))
                    for i in range(batch_start, batch_end):
                        node, state, policy, r = game_buffer[game_buffer_order[i]]

                        pi = policy.weights
                        pi_board = torch.zeros((9, ))
                        pi_board[node.valid_actions] = pi

                        res = evaluator(state.reshape((-1, )))
                        p, v = torch.softmax(res[:9], dim=0), torch.tanh(res[10:])
                        loss_i = (r - v)**2 - torch.sum((pi_board * torch.log(p)))

                        loss += loss_i
                        losses.append(loss_i.item())
                    loss /= (batch_end - batch_start)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(f"Loss: {loss.item():>10.4f} Iteration [{iteration+1:>5.0f}/{iterations:>5.0f}] Batch [{batch_number:>8.0f}/{num_batches:>8.0f}]")
                    on_batch_complete()

            print(f"Iteration [{iteration:>5d}/{iterations}], took {Timer.str('iteration')}")

