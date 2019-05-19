# Neural style transfer
import torch
import torch.nn as nn
import torch.nn.functional as F

# this is an implementation pf JCJhonson and Fei fei li paper version of neural style transfer

class conv2d(nn.Module):
    def __init__(self, nc_in, nc_out, kernel_size, stride, bn=False, activation=False):
        super(conv2d, self).__init__()
        self.bn= bn
        self.activation= activation
        self.conv= nn.Conv2d(nc_in, nc_out, kernel_size, stride)
        if self.bn == True:
            self.norm = nn.BatchNorm2d(nc_out)
    
    def forward(self, x):
        x= self.conv(x)
        if self.bn==True:
            x= self.norm(x)
        if self.activation == True:
            x= F.relu(x)
        return x

class DQN_conv(nn.Module):
    def __init__(self, state_input, action_output, bn_bool=False):
        super(DQN_conv, self).__init__()
        """
        state_input= input frames of the state i.e. no. of input channels of the atari frames
        action_output= no. of output actions := no. of nodes in the final layer
        """
        self.conv1= conv2d(state_input, 32, 8, stride=4, bn=bn_bool, activation=True)
        self.conv2= conv2d(32, 64, 4, stride=2, bn=bn_bool, activation=True)
        self.conv3= conv2d(64, 32, 3, stride=1, bn=bn_bool, activation=True)

        self.fcn1= nn.Linear(32*7*7, 256)
        self.fcn2= nn.Linear(256, action_output) 

    def forward(self, x):
        x= self.conv1(x/255.0) # normalise input
        x= self.conv2(x)
        x= self.conv3(x)
        x= x.view(-1, 32*7*7) # flatten the conv output to the n-dim vector same as fcn1 input
        x= F.relu(self.fcn1(x))
        action_Q_values = self.fcn2(x)

        return action_Q_values

class DQN_fcn(nn.Module):
    def __init__(self, state_input, action_output):
        super(DQN_fcn, self).__init__()
        # 3 layer fully connected neural network
        self.fcn1= nn.Linear(state_input, 32)
        self.fcn2= nn.Linear(32, 32)
        self.fcn4= nn.Linear(32, action_output)
       
    def forward(self, x):
    
        x= F.relu(self.fcn1(x))
        x= F.relu(self.fcn2(x))
        action_Q_values= self.fcn4(x)

        return action_Q_values
