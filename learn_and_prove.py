import numpy as np
import argparse
import configparser
from ddpg_interaction import info, reset, step, maddpg
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import torch

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help='Specify the input config filename')
parser.add_argument('--file', type=str,
                    help='Specify the input-output weights filename')
parser.add_argument('--train', action='store_true',
                    help='run a pre-trainded neural network agent')
parser.add_argument('--random', action='store_true',
                    help='run a tabula rasa agent')
parser.add_argument('--no-graph', action='store_true',
                    help='run a tabula rasa agent')
args = parser.parse_args()

# read configuration file
configParser = configparser.ConfigParser()
configParser.read(args.config)
config = {
    'n_episodes':    int(configParser['PARAMS']['n_episodes']),
    'max_t':         int(configParser['PARAMS']['max_t']),
    'print_every':   int(configParser['PARAMS']['print_every']),
    'SEED':          int(configParser['PARAMS']['SEED']),
    'BUFFER_SIZE':   int(float(configParser['PARAMS']['BUFFER_SIZE'])),
    'BATCH_SIZE':    int(configParser['PARAMS']['BATCH_SIZE']),
    'UPDATE_EVERY':  int(configParser['PARAMS']['UPDATE_EVERY']),
    'GAMMA':         float(configParser['PARAMS']['GAMMA']),
    'SIGMA':         float(configParser['PARAMS']['SIGMA']),
    'TAU':           float(configParser['PARAMS']['TAU']),
    'LR_ACTOR':      float(configParser['PARAMS']['LR_ACTOR']),
    'LR_CRITIC':     float(configParser['PARAMS']['LR_CRITIC']),
    'WEIGHT_DECAY':  float(configParser['PARAMS']['WEIGHT_DECAY']),
    'FC1_ACTOR':     int(configParser['PARAMS']['FC1_ACTOR']),
    'FC2_ACTOR':     int(configParser['PARAMS']['FC2_ACTOR']),
    'FC1_CRITIC':    int(configParser['PARAMS']['FC1_CRITIC']),
    'FC2_CRITIC':    int(configParser['PARAMS']['FC2_CRITIC']),
}
filenames = {
    'game_file':     configParser['FILES']['game_file'],
}
# config configuration
print(' **** Config Parameters')
for k,v in config.items():
    print('{:<15}: {:>15}'.format(k,v))
# files configuration
print(' **** File names')
for k,v in filenames.items():
    print('{:<15}: {:>15}'.format(k,v))

# setting filename
if args.file==None:
    filename='checkpoint'
else:
    filename=args.file

#create environment
env = UnityEnvironment(file_name=filenames['game_file'], seed=config['SEED'])
# get info of the environment
num_agents, state_size, action_size = info(env)

# create two independent agents
agents = [ Agent(num_agents=1, state_size=state_size, action_size=action_size,
                 random_seed=config['SEED'],
                 gamma=config['GAMMA'],
                 sigma=config['SIGMA'],
                 tau=config['TAU'],
                 lr_actor=config['LR_ACTOR'],
                 lr_critic=config['LR_CRITIC'],
                 weight_decay=config['WEIGHT_DECAY'],
                 fc1_a=config['FC1_ACTOR'],
                 fc2_a=config['FC2_ACTOR'],
                 fc1_c=config['FC1_CRITIC'],
                 fc2_c=config['FC2_CRITIC'],
                 buffer_size=config['BUFFER_SIZE'],
                 batch_size=config['BATCH_SIZE'],
                 update_every=config['UPDATE_EVERY']),
           Agent(num_agents=1, state_size=state_size, action_size=action_size,
                 random_seed=config['SEED'],
                 gamma=config['GAMMA'],
                 sigma=config['SIGMA'],
                 tau=config['TAU'],
                 lr_actor=config['LR_ACTOR'],
                 lr_critic=config['LR_CRITIC'],
                 weight_decay=config['WEIGHT_DECAY'],
                 fc1_a=config['FC1_ACTOR'],
                 fc2_a=config['FC2_ACTOR'],
                 fc1_c=config['FC1_CRITIC'],
                 fc2_c=config['FC2_CRITIC'],
                 buffer_size=config['BUFFER_SIZE'],
                 batch_size=config['BATCH_SIZE'],
                 update_every=config['UPDATE_EVERY']) ]

# learn or prove
if args.train:
    scores, scores_avg = maddpg(env, agents,
                                n_episodes=config['n_episodes'],
                                max_t=config['max_t'],
                                print_every=config['print_every'],
                                filename=filename)
    # save training curves
    np.savez(filename+'.npz', scores=scores, scores_avg=scores_avg, config=config)
elif args.random:
    # choose random policy
    print('Run a Tabula Rasa or random agent')
else:
    # load the weights from file
    agents[0].actor_local.load_state_dict(torch.load(filename+'.actor0.pth'))
    agents[0].actor_target.load_state_dict(torch.load(filename+'.actor0.pth'))
    agents[1].critic_local.load_state_dict(torch.load(filename+'.critic1.pth'))
    agents[1].critic_target.load_state_dict(torch.load(filename+'.critic1.pth'))
    print('Loaded {}:'.format(filename))

# run a random controller through arms
if args.no_graph:
    print('no proves on controller')
else:
    for i in range(5):
        scores = np.zeros(len(agents))
        states = reset(env, train_mode=False)
        # check performance of the agent
        #for j in range(1000):
        while True:
            action0 = agents[0].act(states, add_noise=False)
            action1 = agents[1].act(states, add_noise=False)
            actions = np.reshape(np.concatenate((action0,action1),axis=0),(1,4))
            next_states, rewards, dones = step(env, actions)
            states =  next_states
            scores += rewards
            if np.any(dones):
                break
        print('Total score (averaged over agents) this episode: {}'.
              format(np.max(scores)))
env.close()
