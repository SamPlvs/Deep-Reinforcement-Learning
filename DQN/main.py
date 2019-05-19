import torch
import torch.nn as nn
from arguments import *
import numpy as np
from networks import *
from agent import *
from atari_wrappers import * # taken from; OpenAI baselines.common.atari_wrappers
import gym

args= get_args()

def main():

    # declare the environment
    env= make_atari(args.env_name)
    env= wrap_deepmind(env, frame_stack=True)    
    num_actions= env.action_space.n
    num_obs= env.observation_space.shape

    print('State shape: ', num_obs)
    print('Number of actions: ', num_actions)

    # Declare the model
    model= DQN_cnn(state_input= 1, action_output= num_actions)

    agent= DQN_Agent(args, env, model)

    #training the agent:
    if args.train:  
        agent.train()

    if args.test:
        # num of testing trials:
        num_trials= 50

        agent.test(num_trials)

if __name__=='__main__':
    main()