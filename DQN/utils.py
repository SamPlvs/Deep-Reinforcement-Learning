from collections import namedtuple, deque
import random
from arguments import *
import numpy as np
import torch
import torch.nn as nn

args= get_args()
random.seed(args.seed)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if args.initializer== 'orthogonal':
            nn.init.othogonal_(m.weight.data)
        elif args.initializer =='xavier':
            nn.init.xavier_normal_(m.weight.data)
        else:
            nn.init.kaiming_normal_(m.weight.data)
        
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

    if isinstance(m, nn.Linear):
        if args.initializer== 'orthogonal':
            nn.init.othogonal_(m.weight.data)
        elif args.initializer =='xavier':
            nn.init.xavier_normal_(m.weight.data)
        else:
            nn.init.kaiming_normal_(m.weight.data)
        
        if m.bias is not None:
            nn.init.normal_(m.bias.data)



def ToTensor(array):
    if array.ndim == 1:
        tensor= torch.from_numpy(array).float().unsqueeze(0)
    elif array.ndim== 3:
        tensor= np.transpose(array,(2,0,1))
        # 3 dim tensor: C x H x W, so add extra dim for B (batch) to make it: B x C x H x W
        tensor= np.expand_dims(tensor, 0)
        tensor= torch.from_numpy(tensor, dtype= torch.float32)

    elif array.ndim==4:
        tensor= np.transpose(array,(0,3,1,2))
        # no need to expand
        tensor= torch.from_numpy(tensor, dtype= torch.float32)

    return tensor

class ReplayBuffer():
    def __init__(self, arguments):
        self.args= arguments
        self.memory= deque(maxlen= self.args.buffer_size)
        self.experience= namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def write_to_memory(self, state, action, reward, next_state, done):
        # writing the transitions to memory
        if len(self.memory) > self.args.buffer_size:
            self.memory.popleft()
        transition= self.experience(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        # randomly sample a batch of experience 
        experiences= random.sample(self.memory, k= self.args.batch_size)
        # convert numpy tensors to PyTorch tensors!
        states= torch.tensor(np.vstack([i.state for i in experiences if i is not None]), dtype=torch.float32)
        actions= torch.tensor(np.vstack([i.action for i in experiences if i is not None]), dtype=torch.long)
        rewards= torch.tensor(np.vstack([i.reward for i in experiences if i is not None]), dtype=torch.float32)
        next_states= torch.tensor(np.vstack([i.next_state for i in experiences if i is not None]), dtype=torch.float32)
        dones= torch.tensor(np.vstack([i.done for i in experiences if i is not None]), dtype=torch.float32)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class ReplayMemory():
    def __init__(self, arguments):
        self.args= arguments
        self.memory= []
        self.index= 0

    def write_to_memory(self, state, action, reward, next_state, done):
        data= (state, action, reward, next_state, done)
        if self.index >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.index] = data
        # next index:
        self.index= (self.index + 1) % self.args.buffer_size

    def get_random_data(self, idx):
        states, actions, rewards, next_states, dones= [], [], [], [], []
        for i in idx:
            data= self.memory[i]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(np.asarray(action))
            rewards.append(reward) # an int (not an array)
            next_states.append(np.asarray(next_state))
            dones.append(done) # an int (not an array)

        # convert lists to arrays!
        S, A, R, S_, D= np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(next_states), np.asarray(dones)
        S= torch.tensor(S, dtype= torch.float32)
        A= torch.tensor(A, dtype= torch.long)
        R= torch.tensor(R, dtype= torch.float32)
        S_= torch.tensor(S_, dtype= torch.float32)
        D= torch.tensor(D, dtype=torch.float32)
        
        return (S, A, R, S_, D)

    def sample(self):
        random_indexs= [random.randint(0, len(self.memory) - 1) for _ in range(self.args.batch_size)]
        return self.get_random_data(random_indexs)

def ConvOutput_size(input_size, kernel_size, stride, padding, num_layers):
    outputs=[input_size]
    for i in range(num_layers):
        conv_out= ((outputs[i]- kernel_size[i]) + 2*padding[i])//stride[i] + 1
        outputs.append(conv_out)
    return outputs[-1]
