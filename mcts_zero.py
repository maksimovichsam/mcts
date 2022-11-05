from abc import ABC, abstractmethod
from copy import deepcopy
from unicodedata import name
from smaksimovich import Timer
import datetime
import random
import math
import time
import torch
import torch.nn as nn

from smaksimovich import progress, unzip

class SearchPolicy:
    
    def __init__(self, weights: list[float]):
        self.weights = weights
        self.indices = list(range(len(self.weights)))

    def pick(self) -> int:
        return random.choices(population=self.indices, weights=self.weights, k=1)[0]


class MCTSZeroNode(ABC):
    c_puct = 4
    
    def __init__(self):
        self.nodes = None
        self.visits = None
        self.total_value = None
        self.p = None
        self.q = None
        self.v = None

    def node_children(self) -> list['MCTSZeroNode']:
        if self.nodes is not None:
            return self.nodes
        self.nodes = self.children()
        return self.nodes

    def is_terminal(self) -> bool:
        return len(self.node_children()) == 0

    def is_leaf(self) -> bool:
        return self.visits is None

    def select(self) -> tuple[int, 'MCTSZeroNode']:
        total_visits = sum(self.visits)
        puct = lambda i: MCTSZeroNode.c_puct * self.p[i] * math.sqrt(total_visits) / (1 + self.visits[i])
        
        children = self.node_children()
        best_index = 0
        best_value = self.q[0] + puct(0)
        for i in range(1, len(children)):
            value = self.q[i] + puct(i)

            if value > best_value:
                best_value = value
                best_index = i
        return best_index, children[best_index]

    def expand(self) -> float:
        self.p, self.v = self.evaluate(self.state())
        self.visits      = torch.zeros((len(self.p), ), dtype=torch.float)
        self.total_value = torch.zeros((len(self.p), ), dtype=torch.float)
        self.q           = torch.zeros((len(self.p), ), dtype=torch.float)
        return self.v

    def backup(self, child_index: int, value: float) -> None:
        self.total_value[child_index:child_index+1] += value
        self.visits[child_index:child_index+1] += 1
        self.q[child_index:child_index+1] = self.total_value[child_index] / self.visits[child_index]

    def search_policy(self, temperature: float=1.0) -> SearchPolicy:
        coldness = 1 / temperature
        weights = torch.tensor([num_visits**coldness for num_visits in self.visits])
        weights /= torch.sum(weights)
        return SearchPolicy(weights)

    @abstractmethod
    def reset(self):
        """ A static method, deletes the search tree
        """
        pass

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
    def search(node: MCTSZeroNode, simulations:int=100, temperature:float=1.0) -> SearchPolicy:
        root: MCTSZeroNode = node
        for i in range(simulations):
            path: list[tuple[int, MCTSZeroNode]] = []
            while not node.is_leaf() and not node.is_terminal():
                action, new_node = node.select()
                path.append((action, node))
                node = new_node
           
            v = node.expand() if node.is_leaf() else node.reward()
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
    def train_evaluator(root: MCTSZeroNode, evaluator, batch_size=256, iterations=10, num_episodes=100, **kwargs):
        optimizer = torch.optim.AdamW(evaluator.parameters(), lr=evaluator.hp.lr)

        mse_loss = nn.MSELoss()
        num_batches = num_episodes // batch_size

        def compute_loss(path, policies, r):
            # p, v = unzip(node.evaluate(node.state()) for node in path)
            # rewards = [r * (-1)**(idx % 2) for idx in range(len(path) - 1, -1, -1)]
            v = [node.v for node in reversed(path)]
            rewards = [r * (-1)**(idx % 2) for idx in range(len(path))]

            v = torch.stack(v)
            z = torch.tensor(rewards, dtype=torch.float).reshape((-1, 1))

            nll_loss = -1 * torch.sum(
                torch.stack(tuple(
                    torch.dot(policies[i].weights, torch.log(path[i].p + 0.001)) 
                        for i in range(len(path))
                ))
            )

            loss = mse_loss(z, v) + nll_loss
            return loss

        losses = []

        Timer.start("iteration")
        for iteration in range(iterations):

            examples = []

            Timer.start("episode")
            optimizer.zero_grad()
            for episode_i in range(num_episodes):
                root.reset()
                root_copy = deepcopy(root)
                path, policies, r = MCTSZero.play(root_copy, **kwargs)
                assert len(path) == len(policies)
                examples.append((path, policies, r))

                # if progress(episode_i, num_episodes, Timer.elapsed("episode"), percent_to_progress=2):
                #     print(f"Episode [{episode_i:>5d}/{num_episodes}], took {Timer.str('episode')}")
                #     Timer.start("episode")

            # random.shuffle(examples)

            Timer.start("batch")
            for batch_number in range(0, len(examples), batch_size):

                batch = examples[batch_number:batch_number+batch_size]
                batch = torch.stack([compute_loss(path, pi, r) for path, pi, r in batch])
                batch = torch.sum(batch) / batch.shape[0]
                batch.backward()
                optimizer.step()

                # if progress(batch_number // batch_size, num_batches, Timer.elapsed("batches")):
                #     print(f"Loss: {batch.item():>10.4f}  [Batch {batch_number // batch_size + 1:>5d}/{num_batches}], took {Timer.str('batch')}")
                #     Timer.start("batch")

                losses.append(batch.item())

            if progress(iteration, iterations, Timer.elapsed("iteration")):
                print(f"Iteration [{iteration:>5d}/{iterations}], took {Timer.str('iteration')}")
                print(f"Loss: {batch.item():>10.4f}  [Batch {batch_number // batch_size + 1:>5d}/{num_batches}], took {Timer.str('batch')}")
                Timer.start("iteration")

        from matplotlib import pyplot as plt
        plt.plot(list(range(len(losses))), losses)
        plt.show()
