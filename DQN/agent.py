'''
DQN Agent

Sam Tukra

Acknowledgements: Udacity deep reinforcement learning github, TianhongDai github

'''
from datetime import datetime
from time import gmtime, strftime
import os
import numpy as np
import gym
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import *

class DQN_Agent():

    """ 
    1. First random actions are computed for collecting some sample data from the environment. These are stored in replay memory
    2. for every N step iteration, a batch (user defined size) of samples are taken from the replay memory
    3. For each example in the batch / or for each batch as a whole, we calculate the target value (estimate of the state and action):
        Q_value = r_t + gamma * argmax_a Q(s', a)
       and we also calculate the current Q estimate of the state and action (this is the prediction from the NN model):
        Q* = model.predict(Q(s,a))
    4. Calculate the loss i.e. the Temporal difference error: delta:
        delta= Q* - Q_value
       and using the above: huber loss (L): L(delta)
    5. Calculate the gradient of L w.r.t all parameters and update the DQN model
    6. Repeat the steps above till optimal policy convergence
    """

    def __init__(self, arguments, model):
        
        self.args= arguments        
        self.model= model
        # getting the info from the environment
        self.env= gym.make(self.args.env_name) # the environment in which you want to test the agent in. i.e. (gym; 'Atari-breakoutV1')
        self.num_actions= self.env.action_space.n
        # Declaring Memory
        # initialise memory as zeros:
        self.memory= ReplayMemory(self.args) #ReplayBuffer(self.args)

        # Declaring the Q Network
        self.Q_policy_net= self.model # this is the network we deploy
        self.Q_target_net= self.model
        self.Q_target_net.load_state_dict(self.Q_policy_net.state_dict())
        self.optimizer= torch.optim.Adam(self.Q_policy_net.parameters(), self.args.lr)

        if self.args.train:
            if self.args.cuda:
                self.Q_policy_net.cuda();
                self.Q_target_net.cuda();

            self.Q_target_net.eval() # we don't train the target net! we update it by copying the weights from the policy...

        if self.args.test:
            net_dict = torch.load(self.args.load_model)
            self.Q_policy_net.load_state_dict(net_dict)
            if self.args.cuda:
                self.Q_policy_net.cuda();

            self.Q_policy_net.eval()


    # define how actions will be performed intially (uses Epsilon Greedy method): i.t. the Policy
    def select_action(self, state, TimeStep):
        """ We sometimes use the model for selecting an action and sometimes we sample one uniformly, typical method= EpsilonGreedy
        Epsilon Greedy method:
        - Take a random action with probability epsilon
        - Take current best action with probability (1- epsilon)
        """
        epsilon_threshold= self.args.epsilon_end + (self.args.epsilon_start - self.args.epsilon_end) * math.exp(-1. * TimeStep / self.args.epsilon_decay)
        if np.random.random() > epsilon_threshold:
            # the random value is bigger than the epsilon max probability then we use the network to predict
            with torch.no_grad():
                action_values= self.Q_policy_net(state)
                # t.max(1) returns largest column value of each row. Second column on max result is index of where max element
            	# was found, here index = the action! so we pick action with the larger expected reward.
                action= action_values.max(1)[1].view(1,1)
                action= action.cpu().squeeze().numpy()
        else:
            # otherwise we select a random action:
            action= np.random.randint(self.num_actions)

        return action

    # agent learning process, this function performs a single step of optimization, later on I define a train function
    # which optimises the model in the training loop. This is where we get the TD error
    def optimizer_step(self, experience):
        # experience= the batch samples from your replay buffer
        states, actions, rewards, next_states, dones= experience
        
        if self.args.cuda:
            states= states.cuda()
            actions= actions.cuda()
            rewards= rewards.cuda()
            next_states= next_states.cuda()
            dones= dones.cuda()

        # getting the predicted action values from target model:
        Q_next_target= self.Q_target_net(next_states)
        Q_next_target= Q_next_target.detach().max(1)[0]
        
        #Q_next_target= Q_next_target.gather(1,torch.max(Q_next_target,1)[1].unsqueeze(1)).squeeze(1)
        
        # Q_target calculated from the Bellman Equation:
        Q_target= rewards + (self.args.gamma * Q_next_target * (1-dones))
    
        # get the predicted Q values from the policy net (i.e. expected Q values):
        Q_predicted= self.Q_policy_net(states).gather(1,actions.unsqueeze(1))

        # now compute loss:
        if self.args.loss_type == 'huber':
            # huber loss:
            loss= F.smooth_l1_loss(Q_predicted, Q_target.unsqueeze(1)) 
        elif self.args.loss_type == 'mse':
            # mean square error loss:
            loss= F.mse_loss(Q_predicted, Q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self, Policy_net, Target_net, tau, type='hard'):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
        Policy_net: PyTorch model (weights will be copied from)
        Target_net: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
        """
        if type=='soft':
            for target_parameters, policy_parameters in zip(Target_net.parameters(), Policy_net.parameters()):
                target_parameters.data.copy_(tau*policy_parameters.data + (1.0-tau)*target_parameters.data)

        # hard update simply copies the parameters of the policy directly onto the target net, whereas soft update allows some variablitiy
        else:
            for target_parameters, policy_parameters in zip(Target_net.parameters(), Policy_net.parameters()):
                target_parameters.data.copy_(policy_parameters.data)

    def train(self):
        loss=0
        X=[]
        # training loop over each episode
        for i in range(self.args.episodes):
    
            state= np.asarray(self.env.reset())

            # initialise episode reward at 0
            epi_reward= 0.0
            # can set a for loop for maximum time steps i.e. for t in range(max_t):
            for timestep in range(self.args.max_TimeSteps):    
                Scores=[]
                
                state_tensor= torch.tensor(state, dtype= torch.float32)
                if self.args.cuda:
                    state_tensor= state_tensor.cuda()
                # select an action:
                action= self.select_action(state_tensor.unsqueeze(0), timestep)

                # apply the above action in the environment and get the next observations:
                next_state, reward, done, _= self.env.step(action)

                # add the next state to our memory:
                self.memory.write_to_memory(state, action, reward, np.asarray(next_state), done)

                # get the new rewards:
                epi_reward+= reward
                # assign current state as next_state
                state= next_state
                
                Scores.append(epi_reward)
                if done:
                    break
            X.append(np.mean(Scores))
            if i > self.args.learning_starts and i % self.args.learn_frequency==0:
                experience= self.memory.sample()
                loss= self.optimizer_step(experience)
            
            if i % self.args.target_net_update==0:
                # Update the target network:
                self.update_target(self.Q_policy_net, self.Q_target_net, self.args.tau, type='soft')

            if i % self.args.display_interval==0:
                print('[{}], Episodes: {}, Loss: {:.3f}, Reward:{:.2f}'. format(datetime.now(), i, loss, np.mean(Scores)))
                torch.save(self.Q_policy_net.state_dict(), '{}_{}_dqn_{}episodes.pt'.format(self.args.save_path, self.args.env_name, i))
        
                np.save('{}_rewards.npy'.format(self.args.env_name), np.asarray(X))
        
        print('Training Complete! :)')
    
    def test(self, test_episodes = 50):
        # put the model in eval mode so that gradients aren't calculated:        
        for i in range(test_episodes):
            state= np.asarray(self.env.reset())
            self.env.render()
            epi_rewards= 0
            #self.env.render()
            for t in range(self.args.max_TimeSteps):   
                Score= []
                state_tensor = torch.tensor(state, dtype=torch.float32)
                
                if self.args.cuda:
                    state_tensor= state_tensor.cuda()

                with torch.no_grad():
                    action_value= self.Q_policy_net(state_tensor)

                action= torch.argmax(action_value.squeeze()).item()

                next_state, reward, done, _= self.env.step(action)

                epi_rewards += reward

                state= np.asarray(next_state)

                Score.append(epi_rewards)

                if done: 
                    break                

                # for each episode print the reward value:
            print('Episode Reward:{}'. format(np.mean(np.asarray(Score))))

        env.close()
        print('Testing complete! :)')