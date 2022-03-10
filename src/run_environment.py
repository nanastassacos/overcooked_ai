import itertools, math
from re import L
import numpy as np
import torch
import random
import torch.nn.functional as F
import json

from collections import defaultdict, namedtuple, deque

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.model import SAC
from overcooked_ai_py.agents.agent import AgentPair, FixedPlanAgent, GreedyHumanModel, NNPolicy, RandomAgent, SampleAgent, AgentFromPolicy
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

np.random.seed(42)

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

DISPLAY = False

simple_mdp = OvercookedGridworld.from_layout_name('cramped_room')
large_mdp = OvercookedGridworld.from_layout_name('corridor')
scenario_2_mdp = OvercookedGridworld.from_layout_name('scenario2')

with open('src/overcooked_ai_py/agents/config.json', 'r') as JSON:
    config = json.load(JSON)

def run_environment(config):

    episodes = 10

    start_state = OvercookedState(
        [P((8, 1), s),
            P((1, 1), s)],
        {},
        all_orders=scenario_2_mdp.start_all_orders
    )

    env = OvercookedEnv.from_mdp(scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100, info_level=0)

    model0 = SAC(config=config)
    model1 = SAC(config=config)
    
    policy0 = NNPolicy(env, model0)
    policy1 = NNPolicy(env, model1)

    a0 = AgentFromPolicy(policy0)
    a1 = AgentFromPolicy(policy1)

    agent_pair = AgentPair(a0, a1)

    for episode_idx in range(episodes):
        env.reset()
        
        trajectory, timestep, sparse_reward, shaped_reward = env.run_agents(agent_pair)
        
        for i in range(len(trajectory)-1):
            state = env.lossless_state_encoding_mdp(trajectory[i][0])
            state0 = state[0]
            state1 = state[1]

            action0 = Action.ACTION_TO_INDEX[trajectory[i][1][0]]
            action1 = Action.ACTION_TO_INDEX[trajectory[i][1][1]]
            reward = trajectory[i][2]
            done = trajectory[i][3]

            a0.policy.model.save_experience(torch.Tensor(state0)[None, :], action0, reward, float(done))
            a1.policy.model.save_experience(torch.Tensor(state1)[None, :], action1, reward, float(done))
    
        a0.train()
        a1.train()

run_environment(config)
