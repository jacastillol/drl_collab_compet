# Deep RL for Collaborative & Competitive environments
Two DRL Agents learn under a collaborative and competitive environment of a Tennis game using Unity ML-Agents.

## Project Details
The [Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)'s Set-up consist of Two-player game where agents control rackets to bounce ball over a net, in this project we use a little modified game with the following details:

* __Reward Signal:__ If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
* __Observation Space:__ The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
* __Action Space:__ Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). After each episode, the game adds up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started
### Dependencies and Installation
1. Clone this repo `git clone https://github.com/jacastillol/drl_collab_compet.git drl_collab_compet`
1. Create conda environment and install dependencies:
    ```bash
    conda crate --name drlcontinuous python=3.6
    source activate drlcollabcompet
    pip install unityagents torch torchsummary
    ```
1. Download and intall Unity environment
    * Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    * Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    * Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    Then, place the file in the `drl_collab_compet/` folder of this GitHub repository, and unzip the file.

## Instructions
### Configuration File
To run the main program you have to create a configuration file called `*.ini` where you can set all important parameters (e.g, the networks architectures, learning Adam algorithms, replay memory, and others). The repository has an example file `params_example.ini` that you can copy and rename. Here is an example of a configuration file,

    ```python
    [PARAMS]
    # max. number of episode to train the agent
    n_episodes:      2000
    # max. number of steps per episode
    max_t:           1000
    # save the last XXX returns of the agent
    print_every:       50
    # replay buffer size
    SEED:               0
    # replay buffer size
    BUFFER_SIZE:      1e5
    # minibatch size
    BATCH_SIZE:       128
    # how often to update the network
    UPDATE_EVERY:       1
    # discount factor
    GAMMA:           0.99
    # std noise over actions for exploration
    SIGMA:           0.20
    # for soft update or target parameters
    TAU:             6e-2
    # learning rate of the actor
    LR_ACTOR:        1e-4
    # learning rate of the critic
    LR_CRITIC:       1e-3
    # L2 weight decay
    WEIGHT_DECAY:       0
    # number of neurons in actor first layer
    FC1_ACTOR:        400
    # number of neurons in actor second layer
    FC2_ACTOR:        300
    # number of neurons in critic first layer
    FC1_CRITIC:       400
    # number of neurons in critic second layer
    FC2_CRITIC:       300
    [FILES]
    # executable Tenis game. Add path and file
    game_file:       Tennis_Linux/Tennis.x86_64
    ```

### How to run the code

1. Create a config file. One way could be cp params_example.ini params.ini and then modify the parameters as you want

1. Remember to activate the environment with source activate drlnavigation

1. To train a new agent:

    ```bash
    python learn_and_prove.py params.ini --train
    ```

    This will produce three files under the namespace checkpoint: checkpoint.actor.pth and checkpoint.critic.pth holding the weights of the final Actor and Critic networks. The third file checkpoint.npz contains information about the configuration run and learning curves. To change the default namespace use the option --file NAMESPACE.

1. To watch again the performance of the agent trained in the last step run again:

    ```bash
    python learn_and_prove.py [--file NAMESPACE]
    ```