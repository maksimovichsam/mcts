import random, math
import torch
import torch.nn as nn
from mcts_zero import MCTSZero, MCTSZeroNode
from matplotlib import pyplot as plt
from smaksimovich.torch_utils import BasicNN, SimpleDataset
from binary_tree import level_traversal, minimax, pretty_print_tree

hp = BasicNN.HyperParameters()
hp.lr = 0.001
binary_tree_evaluator = BasicNN([1, 100, 100, 100, 100, 3], hp)

class BinaryTreeNode(MCTSZeroNode):

    def __init__(self, index, val=0, left=None, right=None):
        super().__init__()
        self.val = val
        self.left = left
        self.right = right
        self.direction = val
        self.index = index
        self.parent = None

    def children(self):
        res = []
        if self.left:
            res.append(self.left)
        if self.right:
            res.append(self.right)
        return res

    def state(self):
        return torch.tensor([self.index], dtype=torch.float)

    def reward(self):
        return self.val

    def evaluate(self, state) -> tuple[list[float], float]:
        res = binary_tree_evaluator(state)
        return res[:2], res[2:]


def generate_tree(values, levels=1):
    nodes = 2**levels - 1
    count = nodes
    root = BinaryTreeNode(index=0)
    count -= 1

    for idx, node in enumerate(level_traversal(root)):
        node.val = values[idx]
        node.direction = values[idx]
        if count > 0:
            node.left = BinaryTreeNode(nodes - count)
            node.left.parent = node
            count -= 1
            node.right = BinaryTreeNode(nodes - count)
            count -= 1
            node.right.parent = node
    return root


if __name__ == "__main__":
    levels = 4
    tree_values = ([0] * (2**(levels - 1) - 1)) + \
        [1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1]
        # [1, 1, 1, -1] 
        # [random.choice([-1, 1]) for i in range(2**levels - 1)]
    separator = "-" * ((4 * 2**(levels - 1)) - 1)
    root = generate_tree(tree_values, levels)

    def get_node_value(node):
        val = f"{node.val:>5.2f}"
        return val.rjust(5, "_")

    pretty_print_tree(root, levels)
    print(separator)
    pretty_print_tree(root, levels, get_node_value)
    print(separator)

    x = torch.tensor([1,  2, 3, 4, 5, 6,  7], dtype=torch.float).reshape((-1, 1))
    y = torch.tensor([1, -1, 1, 1, 1, 1, -1], dtype=torch.float).reshape((-1, 1))
    ds = SimpleDataset(x, y)

    MCTSZero.train_evaluator(root, binary_tree_evaluator, 200)

    with torch.no_grad():
        x = torch.arange(1, 7, 0.01).reshape((-1, 1))
        plt.plot(x, torch.arctan(binary_tree_evaluator(x)[:, 2]) / (math.pi / 2), 'g.')
        plt.plot(ds.x, ds.y, 'b.')
        plt.show()


    def get_node_value_pred(node):
        val = f"{torch.arctan(binary_tree_evaluator(torch.tensor([[node.index]], dtype=torch.float))[:, 2]).item() / (math.pi / 2):>5.2f}"
        return val.rjust(5, "_")

    pretty_print_tree(root, levels, get_node_value_pred)    
    print(separator)

    def get_mcts_direction(node):
        val = f"{'??'}"
        if node.visits > 0:
            val = f"{node.visits:4.2f}"
        if node.left and node.right:
            if node.left.visits == 0 or node.right.visits == 0:
                val = f"{'?'}"
            else:
                l = node.left.total_reward / node.left.visits
                r = node.right.total_reward / node.right.visits
                val = f"{'L' if l > r else 'R'}"
        return val.rjust(4, "_")

    # pretty_print_tree(root, levels, get_mcts_direction)    
    # print(separator)

    # minimax(root)
    # d = lambda node: f"{node.direction}".rjust(2, "_")
    # pretty_print_tree(root, levels, d)
    # print(separator)

    