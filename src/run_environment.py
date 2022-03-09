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

with open('overcooked_ai_py/agents/config.json', 'r') as JSON:
    config = json.load(JSON)

def run_environment(config):
    model = SAC(config=config)
    policy = NNPolicy(model)

    scenario_2_mdp = OvercookedGridworld.from_layout_name('scenario2')
    mlam = MediumLevelActionManager.from_pickle_or_compute(scenario_2_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)
    a0 = GreedyHumanModel(mlam)
    a1 = AgentFromPolicy(policy)

    start_state = OvercookedState(
        [P((8, 1), s),
            P((1, 1), s)],
        {},
        all_orders=scenario_2_mdp.start_all_orders
    )
    env = OvercookedEnv.from_mdp(scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100)

    done = False    
    while not done:
        # action must be given as (0, 1)

        state = env.lossless_state_encoding_mdp(env.state)

        # action0 = a0.action(env.state)
        action1 = a1.action(torch.Tensor(state[1][None, :]))

        # action0 = Action.ALL_ACTIONS[0]
        # action1 = Action.ALL_ACTIONS[1]

        # s_tp1, r_t, done, info = env.step([action0, action1])

        done = True

run_environment(config)