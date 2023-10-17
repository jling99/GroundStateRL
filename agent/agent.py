import gym
from collections import deque, namedtuple
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.config import CFG
from agent.buffer import Experience

cfg = CFG()

class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        
        self.conv_spin = nn.Conv1d(input_shape[0], cfg.CONV_SPIN_KERNEL)
        self.conv_rep = nn.Conv1d(input_shape[1], cfg.CONV_REP_KERNEL)
        
        self.dense = nn.Linear(cfg.CONV_REP_KERNEL*input_shape[0] + cfg.CONV_SPIN_KERNEL*input_shape[1], cfg.DENSE_HIDDEN)
        
        self.lstm = nn.LSTM(cfg.DENSE_HIDDEN,cfg.LSTM_SIZE)
        
        self.mu = nn.Linear(cfg.LSTM_SIZE, 1)
        self.std = nn.Linear(cfg.LSTM_SIZE, 1)
        self.value = nn.Linear(cfg.LSTM_SIZE, 1)
        
    def forward(self, x):
        x_spin = x
        x_spin = self.conv_spin(x_spin)
        x_spin = nn.Flatten()(x_spin)
        
        x_reps = torch.transpose(x,1,2)
        x_reps = self.conv_rep(x_reps)
        x_reps = nn.Flatten()(x_reps)
        
        x_cat = torch.cat((x_spin, x_reps),0)
        x_rnn = self.lstm(x_cat)
        
        return F.tanh(self.mu(x_rnn)), F.relu(self.std(x_rnn)), self.value(x_rnn)
    
    
class Agent():
    def __init__(self, env, buffer) -> None:
        self.env = env
        self.buffer = buffer
        self.net = Net(env.observation_space.shape[0])
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LEARNING_RATE) #weight_decay = 1e-5
        
        self.beta = cfg.ENTROPY_BETA
        self.clip = cfg.CLIP
        
        self.net_old = Net(env.observation_space.shape[0])
        self.net_old.load_state_dict(self.net.state_dict())
        
        self.epochs = cfg.EPOCHS
        
    def step(self, state, action, reward, done, next_state, info):
            
        exp = Experience(state, action, reward, done, next_state, info)
        self.buffer.append(exp)
        
    def choose(self, state):
        with torch.no_grad():
            state_action_mu, state_action_std,_ = self.net(torch.tensor(state, dtype=torch.float32))
            action = np.random.normal(state_action_mu.detach().numpy()[0,0], state_action_std.detach().numpy()[0,0])
        return action

    def learn(self):

        states, actions, rewards, dones = self.buffer.sample(self.net)
        states_v = torch.tensor(states, dtype=torch.float32)
        actions_v = torch.tensor(actions, dtype=torch.float32)
        rewards_v = torch.tensor(rewards, dtype=torch.float32)
        rewards_v = (rewards_v - rewards_v.mean()) / (rewards_v.std() + 1e-7)
        
        state_old_mu, state_old_std, state_old_val = self.net_old(states_v)
        old_logprobs = self.logprob(state_old_mu, state_old_std, actions_v)
        
        advantages = (-state_old_val[dones].squeeze(-1)+rewards_v[dones]).unsqueeze(-1)
        
        for _ in range(self.epochs):
            
            state_action_mu, state_action_std, state_action_val = self.net(states_v)
            
            loss_value_v = F.mse_loss(state_action_val.squeeze(-1), rewards_v)
            
            # Evaluating old actions and values
            logprobs = self.logprob(state_action_mu, state_action_std, actions_v)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

            loss_entropy = self.beta*(-(torch.log(2*np.pi*state_action_std.clamp(min=0.1))+1)/2).mean()
            
            # final loss of clipped objective PPO
            loss_policy_v = -torch.min(surr1, surr2) + 0.5 * nn.MseLoss()(state_action_val, rewards_v) 
            
            loss = loss_entropy + loss_policy_v + loss_value_v
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.net_old.load_state_dict(self.net.state_dict())
        self.buffer.buffer.clear()
            
    def logprob(self, mu, std, action):
        a = ((mu.clamp(-1,1)-action)**2)/(2*std.clamp(min=0.1))
        b = torch.log(torch.sqrt(2*np.pi*std.clamp(min=0.1)))
        return a-b
        
        