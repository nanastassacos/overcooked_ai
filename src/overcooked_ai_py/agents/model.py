
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
        nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),
    ]
    
    layers.append(nn.Flatten())
    units = 832
    
    for next_units in hidden_dims:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers).apply(initialize_weights_xavier)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, args):
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
        self.hyperparameters = config["hyperparameters"]
        self.alpha = self.hyperparameters["alpha"]

        self.actor = Actor(self.hyperparameters["Actor"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.hyperparameters["Actor"]["lr"], eps=self.hyperparameters["eps"])

        self.critic_0 = Critic(self.hyperparameters["Critic"])
        self.critic_1 = Critic(self.hyperparameters["Critic"])
        self.critic_optimizer_0 = torch.optim.Adam(self.critic_0.parameters(),
                                                 lr=self.hyperparameters["Critic"]["lr"], eps=1e-4)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=self.hyperparameters["Critic"]["lr"], eps=1e-4)

        self.critic_target_0 = Critic(self.hyperparameters["Critic"])
        self.critic_target_1 = Critic(self.hyperparameters["Critic"])

        copy_model_over(self.critic_0, self.critic_target_0)
        copy_model_over(self.critic_1, self.critic_target_1)

        self.memory = ReplayMemory(capacity=10000)

    def save_experience(self, state, action, reward, done):
        self.memory.push((state, action, reward, done))

    def calculate_critic_loss(self, states, actions, rewards, mask):
        with torch.no_grad():
            probs, log_probs = self.actor(states)
            qf0_next_target = self.critic_target_0(states)
            qf1_next_target = self.critic_target_1(states)

            min_qf_next_target = probs * (torch.min(qf0_next_target, qf1_next_target) - self.alpha * log_probs)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)

            next_q_value = rewards + (1.0 - mask) * self.hyperparameters["discount_rate"] * (min_qf_next_target).view(-1)

        qf0 = self.critic_0(states)[torch.arange(len(actions)), actions]
        qf1 = self.critic_1(states)[torch.arange(len(actions)), actions]

        qf0_loss = F.mse_loss(qf0, next_q_value)
        qf1_loss = F.mse_loss(qf1, next_q_value)

        return qf0_loss, qf1_loss

    def calculate_actor_loss(self, states):
        
        probs, log_probs = self.actor(states)
        qf0_pi = self.critic_0(states)
        qf1_pi = self.critic_1(states)
        min_qf_pi = torch.min(qf0_pi, qf1_pi)
        inside_term = self.alpha * log_probs - min_qf_pi
        policy_loss = (probs * inside_term).sum(dim=1).mean()
        log_probs = torch.sum(log_probs * probs, dim=1)

        return policy_loss, log_probs

    def sample_data(self):
        batch = self.memory.sample(batch_size=12)

        states = []
        actions = []
        rewards = []
        masks = []
        for i in range(len(batch)):
            states.append(batch[i].state)
            actions.append(batch[i].action)
            rewards.append(batch[i].reward)
            masks.append(batch[i].done)
        
        states = torch.vstack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        masks = torch.tensor(masks)

        return states, actions, rewards, masks

    def train_critic(self, states, actions, rewards, masks):

        critic_loss_0, critic_loss_1 = self.calculate_critic_loss(states, actions, rewards, masks)
        self.critic_optimizer_0.zero_grad()
        critic_loss_0.backward()
        self.critic_optimizer_0.step()

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

    def train_actor(self, states):
        
        actor_loss, _ = self.calculate_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self):
        states, actions, rewards, masks = self.sample_data()
        self.train_critic(states, actions, rewards, masks)  # unsure how many times I want to train critic vs actor yet.
        self.train_actor(states)