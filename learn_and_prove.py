import argparse
import configparser

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help='Specify the input config filename')
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
