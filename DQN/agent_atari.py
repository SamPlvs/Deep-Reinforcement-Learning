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
from atari_wrappers import*

class DQN_Agent():
    def __init__(self, env, arguments, model):
        
        self.args= arguments        
        self.model= model
        self.env= env
        
        # Declaring Memory
        self.memory= ReplayMemory(self.args) #ReplayBuffer(self.args)

        # Declaring the Q Network
        self.Q_policy_net= self.model # this is the network we deploy
        self.Q_target_net= self.model

        self.Q_policy_net.apply(init_weights)
        self.Q_target_net.apply(init_weights)

        self.Q_target_net.load_state_dict(self,Q_policy_net.state_dict())
        
        # Declare Optimizer
        self.optimizer= torch.optim.Adam(self.Q_policy_net.parameters(), self.args.lr)

        # multi GPU training:
        if self.args.cuda:
            if torch.cuda.device_count() > 1:
                print("using {} GPUs".format(torch.cuda.device_count()))
                self.Q_policy_net = torch.nn.DataParallel(self.Q_policy_net, list(range(torch.cuda.device_count())))
                self.Q_target_net = torch.nn.DataParallel(self.Q_target_net, list(range(torch.cuda.device_count())))
            self.Q_policy_net.cuda();
            self.Q_target_net.cuda();

        self.Q_target_net.eval() # we don't train the target net! we update it by copying the weights from the policy...
    	
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
                action= action_values.max(1)[1].view(1,1)  #np.argmax(action_values.detach().cpu().numpy().squeeze())
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

        # error clipping:
        if self.args.clip_error:
            loss= -1.0 * loss.clamp(-1,1)

        self.optimizer.zero_grad()
        loss.backward()
        # we can also do gradient clipping (optional) -- hacks for RL convergence
        if self.args.clip_grads:
            for param in self.Q_policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
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
            Target_net.load_state_dict(Policy_net.state_dict())

    def train(self):
        loss=0
        # training loop over each episode
        for i in range(self.args.episodes):
    
            state= np.asarray(self.env.reset())

            # initialise episode reward at 0
            epi_reward= 0.0
            # can set a for loop for maximum time steps i.e. for t in range(max_t):
            avg_rewards= []
            for timestep in range(self.args.max_TimeSteps):    
                Scores=[]
                
                state_tensor= torch.tensor(state, dtype= torch.float32)
                if args.cuda:
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
            avg_rewards.append(np.mean(Scores))

            if i > self.args.learning_starts and i % self.args.learn_frequency==0:
                experience= self.memory.sample()
                loss= self.optimizer_step(experience)
            
            if i % self.args.target_net_update==0:
                # Update the target network:
                self.update_target(self.Q_policy_net, self.Q_target_net, self.args.tau, type='hard')

            if i % self.args.display_interval==0:
                print('[{}], Episodes: {}, Loss: {:.3f}, Reward:{:.2f}'. format(datetime.now(), i, loss, np.mean(Scores)))
                torch.save(self.Q_policy_net.state_dict(), '{}_{}_dqn_{}episodes.pt'.format(self.args.save_path, self.args.env_name, i))

        print('Training Complete! :)')

    def train_iter(self):
        pass
    
    def test(self, test_episodes = 50):
        net = torch.load(self.args.load_model)
        self.Q_policy_net.load_state_dict(net)
        # put the model in eval mode so that gradients aren't calculated:
        self.Q_policy_net.eval()
        
        for i in range(test_episodes):
            state= self.env.reset()
            epi_rewards= 0
            self.env.render()
            for t in range(self.args.max_TimeSteps):   
                state = torch.tensor(state, dtype=torch.float32)
                if args.cuda:
                    state= state.cuda
                with torch.no_grad():
                    action_value= self.Q_policy_net(state)

                action= torch.argmax(action_value.squeeze()).item()
                next_state, reward, done, _= self.env.step(action)
                epi_rewards += reward

                if done: 
                    break                
                else:
                    state= next_state
                # for each episode print the reward value:
            print('Episode Reward:{}'. format(epi_rewards))

        self.env.close()
        print('Testing complete! :)')