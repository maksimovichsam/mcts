import random
from mcts import MCTS, MCTSNode
from binary_tree import level_traversal, pretty_print_tree, minimax

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


if __name__ == "__main__":
    levels = 4
    tree_values = ([0] * (2**(levels - 1) - 1)) + \
        [1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1]
        # [1, 1, 1, -1] 
        # [random.choice([-1, 1]) for i in range(2**levels - 1)]
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

    