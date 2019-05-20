# Credit to: TinahongDai: https://github.com/TianhongDai

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
import argparse
import copy

# the arguments
def get_args():
    parser = argparse.ArgumentParser(description='dqn-tutorial')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='the environment name')
    parser.add_argument('--seed', type=int, default=123, help='random seeds')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--random-eps', type=float, default=0.3, help='the param of eps-greedy')
    parser.add_argument('--buffer-size', type=int, default=1e4, help='the replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size for update')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--target-update-step', type=int, default=10, help='frequency update the target network')
    parser.add_argument('--render', action='store_true', help='if render the env')
    # achieve arguments
    args = parser.parse_args()
    return args

# select actions
def select_actions(q, eps):
    action = np.argmax(q) if random.random() > eps else np.random.randint(q.shape[0])
    return action

class network(nn.Module):
    def __init__(self, input_size, num_actions):
        super(network, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.q_value = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_value(x)
        return q_value 

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env)
    # set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    # start to create the network
    net = network(env.observation_space.shape[0], env.action_space.n)
    # set up a target network
    target_net = copy.deepcopy(net)
    # set the optimizer for the network
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # init a replay buffer
    replay_buffer = []
    # reset the environment
    obs = env.reset()
    # ep rewards record the rewards of current episode
    ep_rewards = 0
    total_reward = []
    for t in range(10000):
        if args.render:
            env.render()
        with torch.no_grad():
            # process the inputs, convert them into tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_value = net(obs_tensor)
            action = select_actions(q_value.numpy().squeeze(), args.random_eps)
        # input aciton input the environment
        obs_, reward, done, _ = env.step(action)
        # store the information of this time-step
        if len(replay_buffer) >= args.buffer_size:
            replay_buffer.pop(0)
        replay_buffer.append((obs, action, reward, float(done), obs_))
        # accumulate the rewards
        ep_rewards += reward
        # assign the obs
        obs = obs_
        if done:
            obs = env.reset()
            total_reward.append(ep_rewards)
            ep_rewards = 0
        if t > args.batch_size:
            # start to train the network
            batch_info = random.sample(replay_buffer, args.batch_size)
            # process the batch info
            ob = np.array([sample[0] for sample in batch_info])
            ob_tensor = torch.tensor(ob, dtype=torch.float32)
            a = np.array([sample[1] for sample in batch_info])
            a_tensor = torch.tensor(a, dtype=torch.int64).unsqueeze(-1)
            r = np.array([sample[2] for sample in batch_info])
            r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(-1)
            d = np.array([sample[3] for sample in batch_info]) 
            d_tensor = torch.tensor(1 - d, dtype=torch.float32).unsqueeze(-1)
            ob_ = np.array([sample[4] for sample in batch_info])
            ob_tensor_ = torch.tensor(ob_, dtype=torch.float32)
            # start to do the update
            with torch.no_grad():
                target_q  = target_net(ob_tensor_)
                target_q, _ = torch.max(target_q, dim=1, keepdim=True)
                target_q = r_tensor + args.gamma * target_q * d_tensor
            real_q = net(ob_tensor)
            real_q = real_q.gather(1, a_tensor)
            loss = (target_q - real_q).pow(2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        if t % args.target_update_step == 0:
            target_net.load_state_dict(net.state_dict())
        print('update is {}, reward_mean is: {}'.format(t, np.mean(total_reward[-10:])))
