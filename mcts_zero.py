from abc import ABC, abstractmethod
from smaksimovich import Timer
from torch.distributions import Dirichlet
from collections import deque
import random
import math
import torch
import torch.nn as nn


class SearchPolicy:
    
    def __init__(self, weights: list[float]):
        self.weights = weights
        self.indices = list(range(len(self.weights)))

    def pick(self) -> int:
        return random.choices(population=self.indices, weights=self.weights, k=1)[0]


class MCTSZeroNode(ABC):
    c_puct = 1
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
        noise = Dirichlet(noise * torch.ones((len(self.visits), ))).sample()
        selection = self.q + MCTSZeroNode.c_puct * ((1 - epsilon) * self.p + epsilon * noise) * math.sqrt(self.total_visits + 0.01) / (1 + self.visits)
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
        coldness = 1 / temperature
        # TODO : try doing self.visits**coldness instead of creating new tensor
        weights = torch.tensor([num_visits**coldness for num_visits in self.visits])
        weights /= torch.sum(weights)
        return SearchPolicy(weights)

    @abstractmethod
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
           
            v = node.reward() if node.is_terminal() else node.expand()
            for index, (action, parent) in enumerate(reversed(path)):
                parent.backup(action, v * (-1)**(index % 2))
            
            node = root

        return root.search_policy(temperature=temperature)

    @staticmethod
    def play(node: MCTSZeroNode, **kwargs) -> tuple[list[MCTSZeroNode], list[SearchPolicy]]:
        path = []
        policies: list[SearchPolicy] = []
        while not node.is_terminal():
            path.append(node)
            policy = MCTSZero.search(node, **kwargs)

            node = node.children()[policy.pick()]
            policies.append(policy)
        return path, policies, node.reward()

    @staticmethod
    def train_evaluator(root: MCTSZeroNode, evaluator, 
            num_epochs=5, num_episodes=1, iterations=10, batch_size=256, buffer_size=512, **kwargs):
        from matplotlib import pyplot as plt

        optimizer = torch.optim.AdamW(evaluator.parameters(), lr=evaluator.hp.lr,
            weight_decay=evaluator.hp.weight_decay)
        mse_loss = nn.MSELoss(reduction='sum')

        game_buffer = deque(maxlen=buffer_size)

        for iteration in range(iterations):
            Timer.start("iteration")

            for episodes in range(num_episodes):
                path, policies, r = MCTSZero.play(root, **kwargs)
                assert len(path) == len(policies)
                game_buffer.append((path, policies, r))
                root.reset()

            random.shuffle(game_buffer)

            num_batches = math.ceil(len(game_buffer) / batch_size)
            for epoch in range(num_epochs):
                for batch_number in range(num_batches):
                    optimizer.zero_grad()
                    loss = torch.tensor([0.0])

                    batch_start = batch_number * batch_size
                    batch_end = min(batch_start + batch_size, len(game_buffer))
                    for i in range(batch_start, batch_end):
                        path, policies, r = game_buffer[i]
                        path = [node.evaluate(node.state()) for node in path]
                        rewards = [r * (-1)**(idx % 2) for idx in range(len(path))]
                        values = [node[1] for node in reversed(path)]

                        v = torch.stack(values)
                        z = torch.tensor(rewards, dtype=torch.float).reshape((-1, 1))

                        nll_loss = torch.sum(
                            torch.stack(tuple( 
                                torch.dot(policies[i].weights, torch.log(path[i][0] + 0.001)) 
                                - torch.dot(policies[i].weights, torch.log(policies[i].weights + 0.001)) 
                                    for i in range(len(path))
                            ))
                        )

                        loss += mse_loss(z, v) - nll_loss

                    loss /= (batch_end - batch_start)
                    loss.backward()
                    optimizer.step()

                    print(f"Loss: {loss.item():>10.4f} Batch [{batch_number:>8.0f}/{num_batches:>8.0f}]")
                    if loss.item() < 0.05:
                        break

            print(f"Iteration [{iteration:>5d}/{iterations}], took {Timer.str('iteration')}")

