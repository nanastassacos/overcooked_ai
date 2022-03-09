import itertools, math
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
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

np.random.seed(42)

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

force_compute_large = False
force_compute = True
DISPLAY = False

simple_mdp = OvercookedGridworld.from_layout_name('cramped_room')
large_mdp = OvercookedGridworld.from_layout_name('corridor')
scenario_2_mdp = OvercookedGridworld.from_layout_name('scenario2')

with open('overcooked_ai_py/agents/config.json', 'r') as JSON:
    config = json.load(JSON)

def run_environment(config):

    start_state = OvercookedState(
        [P((8, 1), s),
            P((1, 1), s)],
        {},
        all_orders=scenario_2_mdp.start_all_orders
    )

    env = OvercookedEnv.from_mdp(scenario_2_mdp, start_state_fn=lambda: start_state, horizon=10)
    env.reset()
    
    model0 = SAC(config=config)
    model1 = SAC(config=config)
    
    policy0 = NNPolicy(env, model0)
    policy1 = NNPolicy(env, model1)


    a0 = AgentFromPolicy(policy0)
    a1 = AgentFromPolicy(policy1)

    done = False    
    while not done:

        agent_pair = AgentPair(a0, a1)
        env.run_agents(agent_pair)


run_environment(config)