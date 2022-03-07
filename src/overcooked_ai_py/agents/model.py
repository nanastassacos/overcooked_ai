
import itertools, math
import numpy as np
import torch
import random
from distutils.command.config import config
from torch import nn
import torch.nn.functional as F
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


def create_conv_network(output_dim, hidden_dims=[],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):


    layers = [
        # Defining a 2D convolution layer
        nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Defining another 2D convolution layer
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ]
    units = 64*6
    
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
        print(capacity)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BaseNetwork(nn.Module):

    def __init__(self, params):
        super(BaseNetwork, self).__init__()
        self.params = params
        self.lr = params["lr"]
        self.hidden_dims = params["hidden_dims"]
        self.input_dim = params["input_dim"]
        self.output_dim = params["output_dim"]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Critic(BaseNetwork):
    def __init__(self, params):
        super(Critic, self).__init__(params)
        self.net = create_conv_network(self.output_dim, self.hidden_dims, output_activation=None)

    def forward(self, states):
        q_values = self.net(states)

        return q_values

class Actor(BaseNetwork):
    def __init__(self, params):
        super(Actor, self).__init__(params)
        output_activation = nn.Softmax(dim=1)
        self.net = create_conv_network(self.output_dim, self.hidden_dims, output_activation=output_activation)

    def forward(self, states):
        probs = self.net(states)
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)

        return probs, log_probs
        
def copy_model_over(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())
        
        
class SAC(object):
    def __init__(self, config):
        self.config = config
        
        self.actor = Actor(self.config["params"])

        self.critic_1 = Critic(self.config["params"])
        self.critic_2 = Critic(self.config["params"])

        self.critic_target_1 = Critic(self.config["params"])
        self.critic_target_2 = Critic(self.config["params"])

        copy_model_over(self.critic_1, self.critic_target_1)
        copy_model_over(self.critic_2, self.critic_target_2)

        self.memory = ReplayMemory(capacity=10000)

    def save_experience(self, mdp_tuple):
        self.memory.push(mdp_tuple)

    def calculate_critic_loss(self, states, actions, rewards, mask):
        with torch.no_grad():
            _, probs, log_probs = self.actor(states)
            qf1_next_target = self.critic_target(states)
            qf2_next_target = self.critic_target_2(states)
            min_qf_next_target = probs * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_probs)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = rewards + (1.0 - mask) * self.config.params["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_1(states).gather(1, actions.long())
        qf2 = self.critic_2(states).gather(1, actions.long())

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, states):
        
        _, probs, log_probs = self.actor(states)
        qf1_pi = self.critic_1(states)
        qf2_pi = self.critic_2(states)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_probs - min_qf_pi
        policy_loss = (probs * inside_term).sum(dim=1).mean()
        log_probs = torch.sum(log_probs * probs, dim=1)

        return policy_loss, log_probs

    def train(self):
        pass