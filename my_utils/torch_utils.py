import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from my_utils import flatten
from matplotlib import pyplot as plt


class BasicNN(nn.Module):

    @dataclass
    class HyperParameters:
        momentum     : float = 0.9
        lr           : float = 0.001
        batch_size   : int   = 256
        epochs       : int   = 200
        activation_fn: nn.Module = nn.ReLU


    @staticmethod
    def create_basic_nn(layers: list[int], activation_fn: nn.Module = nn.ReLU) -> nn.Module:
        return nn.Sequential(
            *flatten(
                [
                    (nn.Linear(layers[i - 1], layers[i]), activation_fn()) 
                        for i in range(1, len(layers))
                ]
            )[:-1] # remove last activation fn
        )


    def __init__(self, layers: list[int], hyperparameters: HyperParameters = HyperParameters()):
        super().__init__()
        self.hp = hyperparameters
        self.layers = BasicNN.create_basic_nn(layers, self.hp.activation_fn)

    def forward(self, x):
        return self.layers(x)

    def train_nn(self, train_data, test_data, loss_fn = nn.MSELoss()):
        """ Trains the model on training data
            Arguments:
                train_data [SimpleDataset]: train dataset from DatasetLoader
                test_data  [SimpleDataset]: test dataset from DatasetLoader
            Returns:
                List[float]: a list of training losses from every epoch
                List[float]: a list of test accuracy from every epoch
        """
        print(f"Training BasicNN on {len(train_data)} instances")
        metrics_str = lambda: "(" + ', '.join(f"{100 * metric:.2f}" for metric in evaluation_metrics[-1]) + ")"
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp.lr)
        dataloader = DataLoader(train_data, batch_size=self.hp.batch_size, shuffle=True)
        loss_plot = []
        evaluation_metrics = [self.evaluation_metrics(test_data)]
        print(f"Starting with loss = {metrics_str()} on test data")

        for epoch in range(1, self.hp.epochs + 1):
            for batch, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = loss_fn(self.forward(X), y)
                loss.backward()
                optimizer.step()

                if batch % 10 == 0:
                    print(f"Loss: {loss.item():.4f}  [Batch {batch:>5d}/{(len(train_data) // self.hp.batch_size):>5d}]")

            loss_plot.append(loss.item())
            evaluation_metrics.append(self.evaluation_metrics(test_data))

            if (epoch - 1) % 1 == 0:
                print(f"Epoch {epoch}/{self.hp.epochs}")
                print(f"(accuracy, precision, recall, f1) = {metrics_str()} on test data")

        return loss_plot, evaluation_metrics

    def evaluation_metrics(self, test_data):
        with torch.no_grad():
            return [nn.MSELoss()(self.forward(test_data.x), test_data.y)]

    def save_to_file(self, file_str):
        torch.save(self.state_dict(), file_str)

    def load_from_file(self, file_str):
        self.load_state_dict(torch.load(file_str))


class SimpleDataset:

    def __init__(self, data_x, data_y):
        assert len(data_x) == len(data_y), "Dataset expected same number of X's as Y's"
        self.x = data_x
        self.y = data_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def from_function(f, start, end, n=1000, noise_level=0):
        noise = torch.rand((n, 1)) * noise_level
        x = start + torch.rand((n, 1)) * (end - start)
        y = f(x) + noise
        return SimpleDataset(x, y)


if __name__ == "__main__":
    f0 = lambda x: x**2 + 1 
    f1 = lambda x: sum(x % i for i in range(2, 5 + 1)) 
    f2 = lambda x: sum(torch.cos(x / k) for k in range(1, 5 + 1)) 
    f3 = lambda x: (x + 4) * (x - 1) * (x - 3) * (x - 5) * (x - 10) / 100 # view x = [-5, 11]
    f4 = lambda x: torch.sin(x)

    hp = BasicNN.HyperParameters()
    hp.epochs = 300

    x0, x1 = -5, 5
    ds = SimpleDataset.from_function(f1, x0, x1, n=1000)

    my_nn = BasicNN([1, 100, 100, 100, 1], hp)
    my_nn.train_nn(ds, ds)

    my_nn.eval()
    with torch.no_grad():
        x, _ = torch.sort(ds.x)
        y = my_nn(x)
        plt.plot(x, y, 'g.')
        plt.plot(ds.x, ds.y, 'b.')
        plt.show()