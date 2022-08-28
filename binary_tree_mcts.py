import random
from tkinter.ttk import Separator
from mcts import MCTS, MCTSNode

class BinaryTreeNode(MCTSNode):

    def __init__(self, val=0, left=None, right=None):
        super().__init__()
        self.val = val
        self.left = left
        self.right = right
        self.direction = val

    def children(self):
        res = []
        if self.left:
            res.append(self.left)
        if self.right:
            res.append(self.right)
        return res

    def reward(self):
        return self.val

    def __str__(self):
        return f"BinaryTreeNode(val={self.val}, left={str(self.left) if self.left else 'None'},right={str(self.right) if self.right else 'None'})"


def level_traversal(root):
    if root is None:
        return

    level = [root]
    while len(level) > 0:
        new_level = []
        for node in level:
            yield node
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
        level = new_level


def generate_tree(values, levels=1):
    count = 2**levels - 1
    root = BinaryTreeNode()
    count -= 1

    for idx, node in enumerate(level_traversal(root)):
        node.val = values[idx]
        node.direction = values[idx]
        if count > 0:
            node.left = BinaryTreeNode()
            node.right = BinaryTreeNode()
            count -= 2
    return root


def pretty_print_tree(root, levels, str_fn=lambda node: f"{node.val:>.0f}".rjust(2, '_')):
    nodes = 2**(levels - 1)
    node_width = len(str_fn(root))
    total_width = (node_width + 1) * nodes 
    level = []
    for idx, node in enumerate(level_traversal(root)):
        level.append(node)
        # idx + 2 is a power of 2
        if (idx + 2) & (idx + 1) == 0:
            num_spaces = (total_width // len(level)) - node_width
            print((" " * num_spaces).join(map(str_fn, level)))
            if len(level) != nodes:
                next_num_spaces = (total_width // (len(level) * 2)) - node_width
                branches = "|" + " " * (node_width - 1) + "\\" + "_" * (node_width + next_num_spaces - 1) + " " * (num_spaces - node_width - next_num_spaces)
                print(branches * len(level))
            level.clear()
    

def minimax(root, player=0):
    if root.left is None and root.right is None:
        return root.val

    next_player = (player + 1) % 2
    left = minimax(root.left, player=next_player)
    right = minimax(root.right, player=next_player)
    if (left > right and player == 0) or (left < right and player == 1):
        root.val = left
        root.direction = 'L'
    else:
        root.val = right
        root.direction = 'R'

    return root.val


if __name__ == "__main__":
    levels = 5
    tree_values = ([0] * (2**(levels - 1) - 1)) + \
        [random.choice([-1, 1]) for i in range(2**levels - 1)]
        # [1, 1, 1, -1] 
        # [1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1]
    separator = "-" * ((4 * 2**(levels - 1)) - 1)
    root = generate_tree(tree_values, levels)

    pretty_print_tree(root, levels)
    print(separator)

    def get_node_value(node):
        val = f"{'?'}"
        if node.visits > 0:
            val = f"{node.total_reward / node.visits:5.2f}"
        return val.rjust(5, "_")

    playouts = 100_000
    node = root
    for i in range(playouts):
        MCTS.select(node, num_rollouts=1)

    pretty_print_tree(root, levels, get_node_value)    
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

    pretty_print_tree(root, levels, get_mcts_direction)    
    print(separator)

    minimax(root)
    d = lambda node: f"{node.direction}".rjust(2, "_")
    pretty_print_tree(root, levels, d)
    print(separator)

    