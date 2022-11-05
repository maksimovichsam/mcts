import torch
from torch import nn
from my_utils import torch_utils
from my_utils.utils import Timer

from matplotlib import pyplot as plt

class FunctionOptimizer(nn.Module):
    """ Optimize functions using gradient descent
        ```
        x = torch.randn((10,)) * 5
        y = x * 2 + 1
        f = lambda p: torch.sum((y - (p[0] * x + p[1]))**2) 
        p = torch.randn((2,))
        optimizer = FunctionOptimizer(f, p)
        loss = optimizer.optimize(lr=0.1)

        plt.plot(list(range(len(loss))), loss)
        plt.plot(x, y, 'g.')
        plt.plot(x, p[0] * x + p[1])
        plt.show()
        ```
    """

    def __init__(self, function, theta):
        super().__init__()
        self.function = function
        self.theta = nn.Parameter(theta)

    def optimize(self, max_epochs=100, lr=0.01, **kwargs):
        """ Arguments:
            max_epochs (int, optional): maximum number of epochs. Defaults to 100.
            lr (float, optional): learning rate. Defaults to 0.01.
        Returns:
            list[float]: 
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        losses = []
        timer = Timer()

        epoch = 0
        while epoch < max_epochs:
            epoch += 1

            optimizer.zero_grad()
            loss = self.function(self.theta)
            loss.backward(**kwargs)
            assert not torch.any(torch.isnan(self.theta.grad)), "NaN gradients found"
            optimizer.step()
            losses.append(loss.item())

            if timer.progress(epoch, max_epochs):
                print(f"Epoch {epoch:>5d} f: {losses[-1]:>8.3f}")

        return losses


expo_kernel_p = torch.randn((2, ))
def expo_kernel(x_i, x_j, p=expo_kernel_p):
    assert len(x_i.shape) == 3, f"Expected (N, N, D) matrices, got {x_i.shape}"
    delta = torch.sum(x_i - x_j, axis=-1)**2
    return (p[1]**2) * torch.exp(-0.5 * delta / (p[0]**2 + 0.01))


class GaussianProcess:
    """ Bivariate distribution - https://www.math3d.org/tpb6sbl8Q """

    def __init__(self, kernel=expo_kernel, noise=0.01):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X, y):
        """ Arguments:
            X - torch array of shape (N, D), a matrix of N D-dimensional
                training examples on the X domain
            y - torch array of shape (N, 1), a matrix of N results from 
                some objective function f
        """
        assert len(X.shape) == 2, f"Expected X to be of shape (N, D)" \
            + f", got shape with {len(X.shape)} dimensions"
        assert len(y.shape) == 2 and y.shape[1] == 1, f"Expected y to" \
            + f" be of shape (N, 1) , got shape {y.shape}"

        N, D = X.shape
        self.N, self.D = X.shape
        self.X = X
        self.mu = torch.sum(y) / N
        self.y = y

        constant = self.N / 2 * torch.log(torch.tensor([torch.pi * 2]))
        x_i = torch.tile(X.reshape((1, N, D)), (N, 1, 1))
        x_j = torch.tile(X.reshape((N, 1, D)), (1, N, 1))
        def p_y_given_X(p):
            self.K = self.kernel(x_i, x_j, p) + self.noise**2 * torch.eye(N)
            self.L = torch.linalg.cholesky(self.K)
            self.alpha = torch.linalg.solve(self.L.T, torch.linalg.solve(self.L, self.y - self.mu))
            return -1 * (-0.5 * torch.matmul(y.T, self.alpha) - torch.trace(self.L) - constant)

        fo = FunctionOptimizer(p_y_given_X, expo_kernel_p)
        losses = fo.optimize(lr=0.1, retain_graph=True)
        plt.plot(list(range(len(losses))), losses)
        plt.show()

    def evaluate(self, X, c=0.95):
        """ Arguments:
            X - torch array of shape (N, D), a matrix of N D-dimensional test
                on the X domain
            c - the confidence bound of the variance in [0, 1.0]
            returns:
                y - torch array of shape (N, 1), a matrix of N interpolated 
                    results from the gaussian process 
                variance - torch array of shape (N, 1), a matrix of the confidence
                    bound of y
        """
        assert 0 <= c <= 1.0, f"Expected confidence in [0, 1], got {c}"
        assert self.K is not None, "evaluate was called before fit was"
        assert len(X.shape) == 2, f"Expected X to be of shape (N, D)" \
            + f", got shape with {len(X.shape)} dimensions"

        N_star, D = X.shape
        assert D == self.D, f"Expected X domain to be {self.D} dimensional" \
            + f", got {D}"

        with torch.no_grad():
            # compute f_*
            x_i = torch.tile(self.X.reshape((self.N, 1, D)), (1, N_star, 1))
            x_j = torch.tile(X.reshape((1, N_star, D)), (self.N, 1, 1))
            K_star = self.kernel(x_i, x_j)
            f_star = self.mu + torch.matmul(K_star.T, self.alpha)

            # compute variance
            x_i = X.reshape(1, N_star, D)
            K_star_star = self.kernel(x_i, x_i)
            v = torch.linalg.solve(self.L, K_star)
            variance = K_star_star - torch.sum(v**2, axis=0)
            variance = variance.reshape((N_star, 1))

            return f_star, 2.72 * torch.sqrt(variance)

if __name__ == "__main__":
    f_objective = lambda x: torch.sin(x / 2) + 10 * torch.exp(x**2 / -33) + 20 * torch.exp((x - 33)**2 / -10) + torch.exp(x - 50)

    N = 50
    N_test = 1000
    test_bound = 5
    lo, hi = 0, 53.4

    x_f = torch.arange(lo, hi, (hi - lo) / 100)
    f = f_objective(x_f)

    X = torch_utils.uniform((N, 1), lo, hi).requires_grad_(True)
    y = f_objective(X)
    y += torch_utils.uniform(y.shape, lo=-1, hi=1)

    gp = GaussianProcess(noise=1)
    gp.fit(X, y)

    X_test = torch.arange(lo - test_bound, hi + test_bound, (hi - lo + test_bound * 2) / N_test).reshape((N_test, 1))
    y_test, y_variance = gp.evaluate(X_test)

    from matplotlib import pyplot as plt
    with torch.no_grad():
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x_f, f, 'b')
        ax.plot(X, y, 'g.')

        ax.plot(X_test, y_test, 'r')
        ax.fill_between(
            (X_test).ravel(),
            (y_test - y_variance).ravel(),
            (y_test + y_variance).ravel(),
            color='b', alpha=0.1)

    plt.show()