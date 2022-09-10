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

def pretty_print_tree(root, levels, str_fn=lambda node: f"{node.index:>.0f}".rjust(2, '_')):
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
