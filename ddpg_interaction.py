import time
import numpy as np
from collections import deque
import torch

def info(env):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # examine
    # reset the env
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents =  len(env_info.agents)
    # size of each action
    action_size =  brain.vector_action_space_size
    # examine the state space
    states =  env_info.vector_observations
    state_size =  states.shape[1]
    print('Number of agents:', num_agents)
    print('Size of each action:', action_size)
    print('There are {} agents. Eachobserves a state with length: {}'.
          format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    return num_agents, state_size, action_size

def reset(env,train_mode=True):
    """ Performs an Environment step with a particular action.
    Params
    ======
        env: instance of UnityEnvironment class
    """
    # get the default brain
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    # get state
    states = env_info.vector_observations
    return states

def step(env, actions):
    """ Performs an Environment step with a particular action.
    Params
    ======
        env: instance of UnityEnvironment class
        action: a valid action on the env
    """
    # get the default brain
    brain_name = env.brain_names[0]
    # perform the step
    env_info = env.step(actions)[brain_name]
    # get result from taken action
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    return next_states, rewards, dones
