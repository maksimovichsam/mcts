from abc import ABC, abstractmethod, abstractstaticmethod
import random
import math
import torch
import torch.nn as nn
from binary_tree import pretty_print_tree

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
        self.p = torch.softmax(self.p, dim=0)
        self.v = torch.arctan(self.v) / (math.pi / 2)

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
        pass

    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def evaluate(self, state) -> tuple[list[float], float]:
        pass

    @abstractmethod
    def children(self) -> list['MCTSZeroNode']:
        pass

    @abstractmethod
    def reward(self) -> float:
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
    def train_evaluator(root: MCTSZeroNode, evaluator, num_episodes=100):
        from copy import deepcopy
        optimizer = torch.optim.AdamW(evaluator.parameters(), lr=evaluator.hp.lr)
        
        losses = []

        mse_loss = nn.MSELoss()
        for episode_i in range(num_episodes):
            root.reset()
            optimizer.zero_grad()
            root_copy = deepcopy(root)
            path, policies, r = MCTSZero.play(root_copy)
            assert len(path) == len(policies)

            rewards = [r * (-1)**(idx % 2) for idx in range(len(path))]
            values = [node.v for node in reversed(path)]

            v = torch.stack(values)
            z = torch.tensor(rewards, dtype=torch.float).reshape((-1, 1))

            nll_loss = -1 * torch.sum(
                torch.stack(tuple( 
                    torch.dot(policies[i].weights, torch.log(path[i].p + 0.001)) 
                        for i in range(len(path))
                ))
            )

            loss = mse_loss(z, v) + nll_loss
            loss.backward()
            optimizer.step()

            if episode_i % 10 == 0:
                print(f"Loss: {loss.item():>10.4f}  [Episode {episode_i:>5d}/{num_episodes:>5d}]")

            losses.append(loss.item())

        from matplotlib import pyplot as plt
        plt.plot(list(range(len(losses))), losses)
        plt.show()


# def train_nn(self, train_data, test_data, loss_fn = nn.MSELoss()):
#     """ Trains the model on training data
#         Arguments:
#             train_data [SimpleDataset]: train dataset from DatasetLoader
#             test_data  [SimpleDataset]: test dataset from DatasetLoader
#         Returns:
#             List[float]: a list of training losses from every epoch
#             List[float]: a list of test accuracy from every epoch
#     """
#     print(f"Training BasicNN on {len(train_data)} instances")
#     metrics_str = lambda: "(" + ', '.join(f"{100 * metric:.2f}" for metric in evaluation_metrics[-1]) + ")"
#     optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp.lr)
#     dataloader = DataLoader(train_data, batch_size=self.hp.batch_size, shuffle=True)
#     loss_plot = []
#     evaluation_metrics = [self.evaluation_metrics(test_data)]
#     print(f"Starting with loss = {metrics_str()} on test data")

#     for epoch in range(1, self.hp.epochs + 1):
#         for batch, (X, y) in enumerate(dataloader):
#             optimizer.zero_grad()
#             loss = loss_fn(self.forward(X), y)
#             loss.backward()
#             optimizer.step()

            # if batch % 10 == 0:
            #     print(f"Loss: {loss.item():.4f}  [Batch {batch:>5d}/{(len(train_data) // self.hp.batch_size):>5d}]")

#         loss_plot.append(loss.item())
#         evaluation_metrics.append(self.evaluation_metrics(test_data))

#         if (epoch - 1) % 1 == 0:
#             print(f"Epoch {epoch}/{self.hp.epochs}")
#             print(f"(accuracy, precision, recall, f1) = {metrics_str()} on test data")

#     return loss_plot, evaluation_metrics