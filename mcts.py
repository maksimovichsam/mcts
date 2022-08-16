from abc import ABC, abstractmethod
import math
import random


class MCTSNode(ABC):
    c = math.sqrt(2)
    
    def __init__(self):
        self.visits = 0
        self.total_reward = 0
        self.nodes = None

    def add_reward(self, r, count=1):
        self.total_reward += r
        self.visits += count

    def ucb(self, child_node):
        return child_node.total_reward / child_node.visits \
            + self.c * math.sqrt(math.log(self.visits) / child_node.visits)

    def select(self):
        children = self.node_children()
        best_child = children[0]
        best_value = self.ucb(best_child)
        for i in range(1, len(children)):
            v = self.ucb(children[i])
            if v > best_value:
                best_child = children[i]
                best_value = v
        return best_child

    def expand(self):
        return random.choice(self.node_children())

    def is_leaf(self):
        for child in self.node_children():
            if child.visits == 0:
                return True
        return False

    def is_terminal(self):
        return len(self.node_children()) == 0

    def node_children(self):
        if self.nodes is not None:
            return self.nodes
        self.nodes = self.children()
        return self.nodes

    @abstractmethod
    def children(self):
        pass

    @abstractmethod
    def reward(self):
        pass


class MCTS:

    @staticmethod
    def select(root: MCTSNode, num_rollouts: int=1):
        path = []
        while not root.is_leaf() and not root.is_terminal():
            path.append(root)
            root = root.select()

        path.append(root)
        if root.is_terminal():
            r = root.reward()
            for idx, node in enumerate(path):
                r_i = 0.5 if r == 0 else max(0, r * (-2 * (idx % 2) + 1))
                node.add_reward(r_i)
            return

        root = root.expand()
        path.append(root)

        for i in range(num_rollouts):
            node = root
            while not node.is_terminal():
                node = node.expand()

            r = node.reward()
            for idx, node in enumerate(path):
                r_i = 0.5 if r == 0 else max(0, r * (-2 * (idx % 2) + 1))
                node.add_reward(r_i)

        path.pop()
            
