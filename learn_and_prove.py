import numpy as np
import argparse
import configparser
from ddpg_interaction import info, reset, step, ddpg
from ddpg_agent import Agent

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help='Specify the input config filename')
parser.add_argument('--file', type=str,
                    help='Specify the input-output weights filename')
parser.add_argument('--train', action='store_true',
                    help='run a pre-trainded neural network agent')
args = parser.parse_args()

# read configuration file
configParser = configparser.ConfigParser()
configParser.read(args.config)
config = {
    'n_episodes':    int(configParser['DEFAULT']['n_episodes']),
    'max_t':         int(configParser['DEFAULT']['max_t']),
    'print_every':   int(configParser['DEFAULT']['print_every']),
    'SEED':          int(configParser['DEFAULT']['SEED']),
    'BUFFER_SIZE':   int(float(configParser['DEFAULT']['BUFFER_SIZE'])),
    'BATCH_SIZE':    int(configParser['DEFAULT']['BATCH_SIZE']),
    'UPDATE_EVERY':  int(configParser['DEFAULT']['UPDATE_EVERY']),
    'GAMMA':         float(configParser['DEFAULT']['GAMMA']),
    'SIGMA':         float(configParser['DEFAULT']['SIGMA']),
    'TAU':           float(configParser['DEFAULT']['TAU']),
    'LR_ACTOR':      float(configParser['DEFAULT']['LR_ACTOR']),
    'LR_CRITIC':     float(configParser['DEFAULT']['LR_CRITIC']),
    'WEIGHT_DECAY':  float(configParser['DEFAULT']['WEIGHT_DECAY']),
    'FC1_ACTOR':     int(configParser['DEFAULT']['FC1_ACTOR']),
    'FC2_ACTOR':     int(configParser['DEFAULT']['FC2_ACTOR']),
    'FC1_CRITIC':    int(configParser['DEFAULT']['FC1_CRITIC']),
    'FC2_CRITIC':    int(configParser['DEFAULT']['FC2_CRITIC']),
}
# print configuration
print(' Config Parameters')
for k,v in config.items():
    print('{:<15}: {:>15}'.format(k,v))

# setting filename
if args.file==None:
    filename='checkpoint'
else:
    filename=args.file

#create environment
env = UnityEnvironment(file_name='Tennis/Tennis.x86_64', seed=config['SEED'])
# get info of the environment
num_agents, state_size, action_size = info(env)

# create an agent
agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size,
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
              update_every=config['UPDATE_EVERY'])

# learn or prove
if args.train:
    scores, scores_avg = ddpg(env, agent,
                              n_episodes=config['n_episodes'],
                              max_t=config['max_t'],
                              print_every=config['print_every'],
                              filename=filename)
    # save training curves
    np.savez(filename+'.npz', scores=scores, scores_avg=scores_avg, config=config)
