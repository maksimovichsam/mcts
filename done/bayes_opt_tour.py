import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt


def b(x, offset, height):
    return 2 * height / (1 + (x + offset)**2)

def black_box_function(x):
    offsets = np.arange(1, 19 + 1, 2)
    heights = [1,-1,2,-3,4,0,2,-2,1,2]
    assert len(offsets) == len(heights)

    return sum(b(x, -offsets[i], heights[i]) for i in range(len(offsets)))

# Bounded region of parameter space
pbounds = {'x': (-5, 50)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=4,
    n_iter=20,
)

print(optimizer.max)

x = np.arange(pbounds['x'][0], pbounds['x'][1], 0.01)
y = black_box_function(x)
plt.plot(x, y)
plt.show()
