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

### How to run the code
