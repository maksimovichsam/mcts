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

from torch import tensor
t = tensor([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.,
        0., 1., 0., 0., 0., 1., 0., 1., 0.])
minimax_dataset_cache = {}
def minimax_dataset(node, player, dataset):
    game_str = str(node.game)
    if game_str in minimax_dataset_cache:
        return minimax_dataset_cache[game_str]
    if node.is_terminal():
        return node.reward() * (-1 if player.is_first() else 1)

    r = {}
    for action, child in node.children():
        r[action] = minimax_dataset(child, player.next(), dataset)

    res = 0
    if player.is_first():
        res = max(r.values())
    else:
        res = min(r.values())

    # normalized rewards == action probabilities
    p = [1 if a in r and r[a] == res else 0 for a in range(node.action_space())]
    total_p = sum(p)
    p = [p_i / total_p for p_i in p]
    dataset.append((node.state(), p + [res]))

    minimax_dataset_cache[game_str] = res
    return res

if __name__ == "__main__":
    # Create a tic tac optimal dataset
    # X = game state
    # Y = probability distribution of which move to make

    import ttt_zero
    from ttt_zero import TTTZeroNode
    from tictactoe import TicTacToePlayer, TicTacToe
    from smaksimovich import unzip

    dataset = []
    root = ttt_zero.from_game(TicTacToe())
    player = TicTacToePlayer.X
    minimax_dataset(root, player, dataset)
    X, Y = unzip(dataset)
    assert len(X) == len(Y), f"len(X) = {len(X)}, len(Y) = {len(Y)}"
    print(f'Created dataset with {len(dataset)} points')

    res = []
    for idx in range(len(X)):
        x_i, y_i = X[idx], Y[idx]
        x_i = x_i.reshape((-1,)).tolist()        
        res.append(x_i + y_i)

    with open("tictactoe_solved.csv", "w") as file:
        for line in res:
            file.write(','.join(list(map(str, line))) + '\n')
