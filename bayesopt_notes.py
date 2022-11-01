from matplotlib import pyplot as plt
import numpy as np


def expo_kernel(x_i, x_j):
    assert len(x_i.shape) == 3, f"Expected (N, N, D) matrices, got {x_i.shape}"
    return np.exp(-0.5 * np.sum(x_i - x_j, axis=-1)**2)


class GaussianProcess:
    """ Bivariate distribution - https://www.math3d.org/tpb6sbl8Q
    """

    def __init__(self, kernel=expo_kernel, noise=0):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X, y):
        """ Arguments:
            X - np array of shape (N, D), a matrix of N D-dimensional
                training examples on the X domain
            y - np array of shape (N, 1), a matrix of N results from 
                some objective function f
        """
        assert len(X.shape) == 2, f"Expected X to be of shape (N, D)" \
            + f", got shape with {len(X.shape)} dimensions"
        assert len(y.shape) == 2 and y.shape[1] == 1, f"Expected y to" \
            + f" be of shape (N, 1) , got shape {y.shape}"

        N, D = X.shape
        self.N, self.D = X.shape
        self.X = X
        self.mu = np.sum(y) / N
        self.y = y

        x_i = np.tile(X.reshape((1, N, D)), (N, 1, 1))
        x_j = np.tile(X.reshape((N, 1, D)), (1, N, 1))
        self.K = self.kernel(x_i, x_j) + self.noise**2 * np.identity(N)
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y - self.mu))

    def evaluate(self, X, c=0.95):
        """ Arguments:
            X - np array of shape (N, D), a matrix of N D-dimensional test
                on the X domain
            c - the confidence bound of the variance in [0, 1.0]
            returns:
                y - np array of shape (N, 1), a matrix of N interpolated 
                    results from the gaussian process 
                variance - np array of shape (N, 1), a matrix of the confidence
                    bound of y
        """
        assert 0 <= c <= 1.0, f"Expected confidence in [0, 1], got {c}"
        assert self.K is not None, "evaluate was called before fit was"
        assert len(X.shape) == 2, f"Expected X to be of shape (N, D)" \
            + f", got shape with {len(X.shape)} dimensions"

        N_star, D = X.shape
        assert D == self.D, f"Expected X domain to be {self.D} dimensional" \
            + f", got {D}"

        # compute f_*
        x_i = np.tile(self.X.reshape((self.N, 1, D)), (1, N_star, 1))
        x_j = np.tile(X.reshape((1, N_star, D)), (self.N, 1, 1))
        K_star = self.kernel(x_i, x_j)
        f_star = self.mu + np.matmul(K_star.T, self.alpha)

        # compute variance
        x_i = X.reshape(1, N_star, D)
        K_star_star = self.kernel(x_i, x_i)
        v = np.linalg.solve(self.L, K_star)
        variance = K_star_star - np.sum(v**2, axis=0)
        variance = variance.reshape((N_star, 1))

        return f_star, 2.72 * np.sqrt(variance)

if __name__ == "__main__":
    f_objective = lambda x: np.sin(x / 2) + 10 * np.exp(x**2 / -33) + 20 * np.exp((x - 33)**2 / -10) + np.exp(x - 50)

    N = 10
    N_test = 100
    test_bound = 5
    lo, hi = 0, 53.4

    x_f = np.arange(lo, hi, (hi - lo) / 100)
    f = f_objective(x_f)

    X = np.random.uniform(low=lo, high=hi, size=(N, 1))
    y = f_objective(X)

    gp = GaussianProcess()
    gp.fit(X, y)

    X_test = np.arange(lo - test_bound, hi + test_bound, (hi - lo + test_bound * 2) / N_test).reshape((N_test, 1))
    y_test, y_variance = gp.evaluate(X_test)
    print(y_test.shape, y_variance.shape)

    from matplotlib import pyplot as plt

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