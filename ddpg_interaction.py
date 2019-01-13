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
    states = np.reshape(states, (1,48))
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
    next_states = np.reshape(env_info.vector_observations,(1,48))
    rewards = env_info.rewards
    dones = env_info.local_done
    return next_states, rewards, dones

def maddpg(env, agents, n_episodes=300, max_t=700, print_every=10, filename='checkpoint'):
    """ Deep Deterministic Policy Gradient algorithm
    Params:
    =======
    """
    first_time=True
    scores_deque = deque(maxlen=100)
    scores_avg = deque(maxlen=n_episodes)
    scores = []
    max_score = -np.Inf
    # init timer
    tic = time.clock()
    # for each episode
    for i_episode in range(1, n_episodes+1):
        states = reset(env, train_mode=True)
        agents[0].reset()
        agents[1].reset()
        score = np.zeros(len(agents))
        # for t in range(max_t):
        while True:
            action0 = agents[0].act(states)
            action1 = agents[1].act(states)
            actions = np.reshape(np.concatenate((action0,action1),axis=0),(1,4))
            next_states, rewards, dones = step(env, actions)
            agents[0].step(states, actions, rewards[0], next_states, dones, 0)
            agents[1].step(states, actions, rewards[1], next_states, dones, 1)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        score = np.max(score)
        scores_deque.append(score)
        scores.append(score)
        # geting averages
        # append to average
        curr_avg_score = np.mean(scores_deque)
        scores_avg.append(curr_avg_score)
        # update best average reward
        if curr_avg_score > max_score:
            max_score = curr_avg_score
        # monitor progress
        message = "\rEpisode {:>4}/{:>4} || Score {:.5f} || Last avg. scores {:.5f} || Best avg. score {:.5f} "
        if i_episode % print_every == 0:
            print(message.format(i_episode, n_episodes, score, curr_avg_score, max_score))
        else:
            print(message.format(i_episode, n_episodes, score, curr_avg_score, max_score), end="")
        # stopping criteria
        if curr_avg_score>=0.5:
            # save solved weights for
            torch.save(agents[0].actor_local.state_dict(), filename+'_solved.actor0.pth')
            torch.save(agents[0].critic_local.state_dict(), filename+'_solved.critic0.pth')
            torch.save(agents[1].actor_local.state_dict(), filename+'_solved.actor1.pth')
            torch.save(agents[1].critic_local.state_dict(), filename+'_solved.critic1.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tin {:.2f} secs'.
                  format(i_episode, curr_avg_score, time.clock()-tic))
            # break
    # save final weights
    torch.save(agents[0].actor_local.state_dict(), filename+'.actor0.pth')
    torch.save(agents[0].critic_local.state_dict(), filename+'.critic0.pth')
    torch.save(agents[1].actor_local.state_dict(), filename+'.actor1.pth')
    torch.save(agents[1].critic_local.state_dict(), filename+'.critic1.pth')

    return scores, scores_avg
