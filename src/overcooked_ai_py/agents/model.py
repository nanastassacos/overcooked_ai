
import itertools, math
import numpy as np
import torch
import random
from distutils.command.config import config
from torch import nn
from torch.distributions import Normal, Categorical
from collections import defaultdict, namedtuple, deque
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import NNPolicy


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_linear_network(input_dim, output_dim, hidden_dims=[],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):
    assert isinstance(input_dim, int) and isinstance(output_dim, int)
    assert isinstance(hidden_dims, list) or isinstance(hidden_dims, list)

    layers = []
    units = input_dim
    for next_units in hidden_dims:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers).apply(initialize_weights_xavier)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BaseNetwork(nn.Module):

    def __init__(self, params):
        self.params = params
        self.lr = params.lr
        self.hidden_dims = params.hidden_dims
        self.input_dim = params.input_dim
        self.output_dim = params.output_dims

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Critic(BaseNetwork):
    def __init__(self, params):
        BaseNetwork.__init__(self, params)

    def forward(self, states):
        q_values = self.net(states)

        return q_values

class Actor(BaseNetwork):
    def __init__(self, params):
        BaseNetwork.__init__(self, params)
        self.output_activation = nn.LogSoftmax(dim=1)
        self.net = create_linear_network(self.input_dim, self.output_dim, self.hidden_dims, output_activation=self.output_activation)

    def forward(self, states):
        log_probs = self.net(states)
        probs = torch.exp(log_probs)
        max_prob_action = torch.argmax(probs, dim=-1)
        action_dist = Categorical(probs)
        action = action_dist.sample()

        return action_dist, log_probs, action, max_prob_action
        

class SAC(object):
    def __init__(self, config):
        self.config = config
        
        self.actor = Actor(self.config.params)
        self.critic1 = Critic(self.config.params)
        self.critic2 = Critic(self.config.params)
        
        self.memory = ReplayMemory(maxlen=1e5)

    def save_experience(self, mdp_tuple):
        self.memory.push(mdp_tuple)

    