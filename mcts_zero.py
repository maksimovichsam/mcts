from abc import ABC, abstractmethod
from smaksimovich import Timer, unzip
from torch.distributions import Dirichlet
from collections import deque, namedtuple
import random
import math
import torch
from typing import Optional
from torch.nn import functional as F

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds] = None):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value
  

class ExponentialMovingAverage:

    def __init__(self, alpha, initial_value=None):
        assert 0 <= alpha <= 1.0, f"Expected alpha between 0 and 1"
        self.x = initial_value
        self.alpha = alpha

    def add_value(self, value):
        if self.x is None:
            self.x = value
        else:
            self.x = (1 - self.alpha) * self.x + self.alpha * value

EMA = ExponentialMovingAverage


class SearchPolicy:
    
    def __init__(self, weights: list[float]):
        self.weights = weights
        self.indices = list(range(len(self.weights)))

    def pick(self) -> int:
        return random.choices(population=self.indices, weights=self.weights, k=1)[0]


class MCTSZeroNode(ABC):
    c_puct = 2
    noise_scale = 0.10
    
    def __init__(self):
        self.reset()

    def node_children(self) -> list[tuple[float, 'MCTSZeroNode']]:
        if self.nodes is not None:
            return self.nodes
        self.nodes = self.children()
        return self.nodes

    def is_terminal(self) -> bool:
        return len(self.node_children()) == 0

    def is_leaf(self) -> bool:
        return self.visits is None

    def select(self, min_max_stats, gamma, noise=None, epsilon=0.25) -> tuple[int, 'MCTSZeroNode']:
        # rewards = torch.tensor([r for r, child in self.node_children()], dtype=torch.float)
        # q_score = (self.visits > 0) * min_max_stats.normalize(rewards + gamma * self.q)
        q_score = self.q

        # print("selection:\n", self.game)
        # print("q: ", q_score)
        # print("p: ", self.p)
        # print("v: ", self.v)
        # print("visits: ", self.visits)
        if noise is None:
            # print("p_score: ", MCTSZeroNode.c_puct * self.p * math.sqrt(self.total_visits + 1e-8) / (1 + self.visits))
            selection = q_score + MCTSZeroNode.c_puct * self.p * math.sqrt(self.total_visits + 1e-8) / (1 + self.visits)
        else:
            noise = Dirichlet(noise * torch.ones((len(self.visits), ))).sample()
            # print("noise: ", noise)
            # print("p_score: ", MCTSZeroNode.c_puct * ((1 - epsilon) * self.p + epsilon * noise) * math.sqrt(self.total_visits + 1e-8) / (1 + self.visits))
            selection = q_score + MCTSZeroNode.c_puct * ((1 - epsilon) * self.p + epsilon * noise) * math.sqrt(self.total_visits + 1e-8) / (1 + self.visits)
        # print("score: ", selection)
        action = torch.argmax(selection, dim=0)
        reward, child_node = self.node_children()[action]
        # print("action, reward: ", action, reward)
        return action, reward, child_node

    def expand(self) -> float:
        self.p, self.v = self.evaluate(self.state())
        self.visits      = torch.zeros((len(self.p), ), dtype=torch.float)
        self.total_value = torch.zeros((len(self.p), ), dtype=torch.float)
        self.q           = 0 * torch.ones((len(self.p), ), dtype=torch.float)
        return self.v

    def backup(self, child_index: int, value: float, min_max_stats) -> None:
        self.total_value[child_index:child_index+1] += value
        self.visits[child_index:child_index+1] += 1
        self.total_visits += 1
        self.q[child_index:child_index+1] = self.total_value[child_index] / self.visits[child_index]
        # min_max_stats.update(self.q[child_index:child_index+1].item())

    def search_policy(self, temperature: float=1) -> SearchPolicy:
        if temperature == 0:
            weights = torch.zeros_like(self.visits)
            weights[torch.argmax(self.visits, dim=0)] = 1
        else:
            weights = torch.pow(self.visits, 1 / temperature)
            weights /= torch.sum(weights)
        return SearchPolicy(weights)

    def reset(self):
        """ A deletes the search tree and resets nodes statistics"""
        self.visits = None
        self.total_visits = 0
        self.total_value = None
        self.p = None
        self.q = None
        self.v = None
        self.nodes = None

    @abstractmethod
    def action_space(self) -> int:
        """ Returns the maximum number of actions of the game, over all game states
        """
        pass

    @abstractmethod
    def legal_actions(self) -> list[int]:
        """ Returns a list of actions, indicating allowed actions for the given state
        """
        pass

    @abstractmethod
    def state(self):
        """ Returns the tensor representation of this node
        """
        pass

    @abstractmethod
    def evaluate(self, state) -> tuple[list[float], float]:
        """ Computes tensors p, v for the given state
            - p = normalized probabilities of legal moves
            - v = value of the state
        """
        pass

    @abstractmethod
    def children(self) -> list['MCTSZeroNode']:
        """ Returns the sub states after taking one action from 
            the given state
        """
        pass

    @abstractmethod
    def reward(self) -> float:
        """ Returns the game reward from the perspective of the player
            which made the last move
        """
        pass


class AveragingGameBuffer:

    def __init__(self, alpha):
        self.states = {}
        self.alpha = alpha
        self.values = None

    def __getitem__(self, idx):
        return (self.values[idx][0], self.values[idx][1].x, self.values[idx][2].x)

    def __len__(self):
        return len(self.states)

    def add_examples(self, examples):
        for node, state, policy, r in examples:
            key = str(node.game)
            if key not in self.states:
                self.states[key] = [state, EMA(self.alpha, policy), EMA(self.alpha, r)]
            else:
                self.states[key][1].add_value(policy)
                self.states[key][2].add_value(r)
        self.values = list(self.states.values())


class DequeGameBuffer:
    
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

    def add_examples(self, examples):
        for node, state, policy, r in examples:
            self.buffer.append((state, policy, r))

class MCTSZero:

    @staticmethod
    def search(node: MCTSZeroNode, min_max_stats, gamma=0.997, simulations:int=100, temperature:float=1, root_noise=None) -> SearchPolicy:
        root: MCTSZeroNode = node
        debug = False
        for i in range(simulations):
            path: list[tuple[int, MCTSZeroNode]] = []
            while not node.is_leaf() and not node.is_terminal():
                action, reward, new_node = node.select(min_max_stats, gamma=gamma, noise=root_noise if node == root else None)
                path.append((action, reward, node))
                if debug:
                    print(node.game)
                if len(path) > 1000:
                    assert False, "Infinite loop"
                node = new_node
           
            R = node.v if node.v is not None else node.expand()
            for index, (action, reward, parent) in enumerate(reversed(path)):
                R = reward + R * gamma
                parent.backup(action, R, min_max_stats)

            node = root

        return root.search_policy(temperature=temperature)

    @staticmethod
    def play(node: MCTSZeroNode, best_reward, hp, gamma=0.997, **kwargs) -> tuple[list[MCTSZeroNode], list[SearchPolicy]]:
        assert len(node.node_map) == 1
        examples = []
        rewards = []
        episode_step = 0

        min_max_stats = MinMaxStats()
        temperature_threshold = float('inf') if not hasattr(hp, "temperature_threshold") else hp.temperature_threshold
        temperature_threshold = float('inf') if temperature_threshold is None else temperature_threshold
        while not node.is_terminal() and episode_step < 400:
            tau = 1.0 if episode_step < temperature_threshold else 0.0
            policy = MCTSZero.search(node, min_max_stats, temperature=tau, gamma=gamma, **kwargs)

            pi = torch.zeros((node.action_space(), ))
            pi[node.legal_actions()] = policy.weights

            v = torch.sum((node.visits / node.total_visits) * node.q)
            examples.append([node, node.state(), pi, v.item()])

            action = policy.pick()
            reward, node = node.node_children()[action]
            rewards.append(reward)
            episode_step += 1

        # R = 0
        # for idx, reward in enumerate(reversed(rewards)):
        #     R = rewards[len(rewards) - 1 - idx] + R * gamma
        #     rewards[len(rewards) - 1 - idx] = R
        # best_reward = max(best_reward, max(rewards))

        # for idx, example in enumerate(reversed(examples)):
        #     R = rewards[len(rewards) - 1 - idx]
        #     example[-1] = R / best_reward if R > 0 else R

        # R = 0
        # for idx, example in enumerate(reversed(examples)):
        #     R = rewards[len(rewards) - 1 - idx] + gamma * R
        #     example[-1] = R

        # for idx, example in enumerate(reversed(examples)):
        #     node = example[0]
        #     v = torch.sum((node.visits / node.total_visits) * node.q)
        #     example[-1] = v

        return examples, sum(rewards)
        
    @staticmethod
    def train_evaluator(root: MCTSZeroNode, evaluator, 
            num_epochs=5, num_episodes=1, iterations=10, batch_size=256, 
            buffer_size=512, c_puct=4, on_batch_complete=lambda:0, **kwargs):
        MCTSZeroNode.c_puct = c_puct
        hp = evaluator.hp

        optimizer = torch.optim.AdamW(evaluator.parameters(), lr=evaluator.hp.lr,
            weight_decay=hp.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer
            , step_size=hp.lr_decay_steps if hasattr(hp, 'lr_decay_steps') else 1000
            , gamma=hp.gamma if hasattr(hp, 'gamma') else 0.1
            , verbose=True)

        if hasattr(hp, 'alpha') and hp.alpha is not None:
            game_buffer = AveragingGameBuffer(hp.alpha)
        else:
            game_buffer = DequeGameBuffer(buffer_size)

        A = root.action_space()

        losses = []
        rewards = []
        best_reward = 0
        for iteration in range(iterations):
            Timer.start("iteration")

            for episodes in range(num_episodes):
                examples, R = MCTSZero.play(root, best_reward, hp, **kwargs)
                game_buffer.add_examples(examples)
                best_reward = max(best_reward, R)
                rewards.append(R)
                root.reset()
                root = root.clear_game_tree()

            game_buffer_order = list(range(len(game_buffer)))
            num_batches = math.ceil(len(game_buffer) / batch_size)
            evaluator.train()

            for epoch in range(num_epochs):
                random.shuffle(game_buffer_order)

                total_loss = 0
                for batch_number in range(num_batches):
                    batch_start = batch_number * batch_size
                    batch_end = min(batch_start + batch_size, len(game_buffer))

                    states, policies, r = unzip(game_buffer[game_buffer_order[i]] for i in range(batch_start, batch_end))
                    states = torch.concat(states)
                    policies = torch.stack(policies)
                    r = torch.tensor(r, dtype=torch.float).reshape((-1, 1))
                    assert not r.requires_grad
                    assert not policies.requires_grad

                    model_output = evaluator(states)
                    # p, v = torch.softmax(model_output[:, :A], dim=1), torch.tanh(model_output[:, A:]) 
                    # p, v = torch.softmax(model_output[:, :A], dim=1), torch.sigmoid(model_output[:, A:]) 
                    p, v = torch.softmax(model_output[:, :A], dim=1), torch.relu(model_output[:, A:] + 1) - 1
                    assert p.requires_grad
                    assert v.requires_grad

                    # mse loss + nll loss
                    loss = torch.sum((r - v)**2) - torch.sum(policies * torch.log(p + 1e-8)) + torch.sum(policies * torch.log(policies + 1e-8))
                    loss /= (batch_end - batch_start)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    on_batch_complete()
                    total_loss += loss.item()

                total_loss /= num_batches
                losses.append(total_loss)
                print(f"Loss: {total_loss:>10.4f} Iteration [{iteration+1:>5.0f}/{iterations:>5.0f}] Batch [{batch_number:>8.0f}/{num_batches:>8.0f}]")

            avg_R = sum(rewards[-num_episodes:]) / num_episodes
            print(f"Iteration [{iteration+1:>5d}/{iterations}], took {Timer.str('iteration')}, best reward: {best_reward}, average reward: {avg_R}")
            lr_scheduler.step()

        return losses, rewards
