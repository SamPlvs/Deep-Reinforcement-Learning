import torch
import torch.nn as nn
from arguments import *
import numpy as np
from networks import *
from agent import *
import gym

args= get_args()

def main():

    env= gym.make(args.env_name)
    env.seed(0)
    num_actions= env.action_space.n
    num_obs= env.observation_space.shape

    print('State shape: ', num_obs[0])
    print('Number of actions: ', num_actions)

    # Declare the model
    model= DQN_fcn(state_input= num_obs[0], action_output= num_actions)

    agent= DQN_Agent(args, model)

    #training the agent:
    if args.train:  
        agent.train()

    if args.test:
        # num of testing trials:
        num_trials= 50

        agent.test(num_trials)


if __name__=='__main__':
    main()