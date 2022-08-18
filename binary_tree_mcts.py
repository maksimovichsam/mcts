import random
from tkinter.ttk import Separator
from mcts import MCTS, MCTSNode

class BinaryTreeNode(MCTSNode):

    def __init__(self, val=0, left=None, right=None):
        super().__init__()
        self.val = val
        self.left = left
        self.right = right

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


def generate_tree(levels=1):
    count = 2**levels - 1
    root = BinaryTreeNode()
    count -= 1

    for node in level_traversal(root):
        if count <= 0:
            return root
            
        node.left = BinaryTreeNode()
        node.right = BinaryTreeNode()
        count -= 2

        if count < 2**(levels - 1):
            node.left.val = random.choice([-1, 0, 1])
            node.right.val = random.choice([-1, 0, 1])


if __name__ == "__main__":
    levels = 5
    separator = "-" * ((4 * 2**(levels - 1)) - 1)
    root = generate_tree(levels)
    for idx, node in enumerate(level_traversal(root)):
        # idx + 2 is a power of 2
        if (idx + 2) & (idx + 1) == 0:
            print(f"{node.val:>2}")
        else:
            print(f"{node.val:>2}", end=', ')
    print(separator)

    import pickle
    
    save_file = "ttt.pickle"

    playouts = 10_000
    for i in range(playouts):
        MCTS.select(root, num_rollouts=1)

    for idx, node in enumerate(level_traversal(root)):
        val = f"{node.total_reward / node.visits:5.2f}"
        if (idx + 2) & (idx + 1) == 0:
            print(val)
        else:
            print(val, end=', ')
    
    print(separator)

    for idx, node in enumerate(level_traversal(root)):
        val = f"{'??':>5}"
        if node.visits > 0:
            val = f"{node.total_reward / node.visits:5.2f}"
        if node.left and node.right:
            if node.left.visits == 0 or node.right.visits == 0:
                val = f"{'?':>5}"
            else:
                l = node.left.total_reward / node.left.visits
                r = node.right.total_reward / node.right.visits
                val = f"{'L' if l > r else 'R':>5}"

        if (idx + 2) & (idx + 1) == 0:
            print(val)
        else:
            print(val, end=', ')

    
            