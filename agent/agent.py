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

class NetRnn(nn.Module):
    def __init__(self, input_shape):
        super(NetRnn, self).__init__()

        self.conv_spin = nn.Conv1d(input_shape[0], cfg.CONV_SPIN_KERNEL, input_shape[0], padding='same')
        self.conv_rep = nn.Conv1d(input_shape[1], cfg.CONV_REP_KERNEL, input_shape[1], padding='same')
        
        self.dense = nn.Linear(cfg.CONV_REP_KERNEL*input_shape[0] + cfg.CONV_SPIN_KERNEL*input_shape[1], cfg.DENSE_HIDDEN)
        
        self.lstm = nn.LSTM(cfg.DENSE_HIDDEN,cfg.LSTM_SIZE)
        
        self.mu = nn.Linear(cfg.LSTM_SIZE, 1)
        self.std = nn.Linear(cfg.LSTM_SIZE, 1)
        self.value = nn.Linear(cfg.LSTM_SIZE, 1)
        
    def forward(self, x):
        x_spin = x
        x_spin = self.conv_spin(x_spin)
        x_spin = x_spin.reshape(x_spin.shape[0],-1)

        x_reps = torch.transpose(x,1,2)
        x_reps = self.conv_rep(x_reps)
        x_reps = x_reps.reshape(x_reps.shape[0],-1)
        
        x_cat = torch.cat((x_spin, x_reps),1)
        x_cat = self.dense(x_cat)

        x_rnn, _= self.lstm(x_cat)
        
        return F.tanh(self.mu(x_rnn)), F.relu(self.std(x_rnn)), self.value(x_rnn)
    
class NetConv(nn.Module):
    def __init__(self, input_shape):
        super(NetConv, self).__init__()

        self.conv_spin = nn.Conv1d(input_shape[0], cfg.CONV_SPIN_KERNEL, cfg.SIZE_SPIN_KERNEL, padding='same')
        self.conv_rep = nn.Conv1d(input_shape[1], cfg.CONV_REP_KERNEL, cfg.SIZE_REP_KERNEL, padding='same')
        
        self.dense = nn.Linear(cfg.CONV_REP_KERNEL*input_shape[0] + cfg.CONV_SPIN_KERNEL*input_shape[1], cfg.DENSE_HIDDEN)
        
        self.linear = nn.Linear(cfg.DENSE_HIDDEN,cfg.DENSE_HIDDEN)
        
        self.mu = nn.Linear(cfg.DENSE_HIDDEN, 1)
        self.std = nn.Linear(cfg.DENSE_HIDDEN, 1)
        self.value = nn.Linear(cfg.DENSE_HIDDEN, 1)
        
    def forward(self, x):
        x_spin = x
        x_spin = self.conv_spin(x_spin)
        x_spin = x_spin.reshape(x_spin.shape[0],-1)

        x_reps = torch.transpose(x,1,2)
        x_reps = self.conv_rep(x_reps)
        x_reps = x_reps.reshape(x_reps.shape[0],-1)
        
        x_cat = torch.cat((x_spin, x_reps),1)
        x_cat = self.dense(x_cat)

        x_l= self.linear(x_cat)
        
        return F.tanh(self.mu(x_l)), F.relu(self.std(x_l)), self.value(x_l)
    
    
class Agent():
    def __init__(self, env, buffer, recurrent = True) -> None:
        
        self.env = env
        self.buffer = buffer
        
        if recurrent:
            self.net = NetRnn(env.observation_space.shape)
            self.net_old = NetRnn(env.observation_space.shape)
        else :
            self.net = NetConv(env.observation_space.shape)
            self.net_old = NetConv(env.observation_space.shape)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LEARNING_RATE) #weight_decay = 1e-5
        
        self.beta = cfg.ENTROPY_BETA
        self.clip = cfg.CLIP
        
        
        self.net_old.load_state_dict(self.net.state_dict())
        
        self.epochs = cfg.EPOCHS
        
    def step(self, state, action, reward, done, next_state, info):
            
        exp = Experience(state, action, reward, done, next_state, info)
        self.buffer.append(exp)
        
    def choose(self, state):
        with torch.no_grad():
            state_action_mu, state_action_std,_ = self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = np.random.normal(state_action_mu.detach().numpy()[0,0], state_action_std.detach().numpy()[0,0])
        return action

    def learn(self):
        with torch.no_grad():
            states, actions, rewards, dones = self.buffer.sample(self.net)
            states_v = torch.tensor(states, dtype=torch.float32)
            actions_v = torch.tensor(actions, dtype=torch.float32)
            rewards_v = torch.tensor(rewards, dtype=torch.float32)
            rewards_v = (rewards_v - rewards_v.mean()) 
            dones_v = torch.tensor(dones, dtype=torch.bool)
            
            state_old_mu, state_old_std, state_old_val = self.net_old(states_v)
            
            old_logprobs = self.logprob(state_old_mu, state_old_std, actions_v)
            
            advantages = (-state_old_val[dones_v].squeeze(-1)+rewards_v[dones_v]).unsqueeze(-1)

        for _ in range(self.epochs):

            state_action_mu, state_action_std, state_action_val = self.net(states_v)
            

            
            with torch.no_grad():
                logprobs = self.logprob(state_action_mu, state_action_std, actions_v)

                ratios = torch.exp(logprobs - old_logprobs)[dones_v]

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages
                
            loss_entropy = self.beta*(-(torch.log(2*np.pi*state_action_std.clamp(min=0.1))+1)/2).mean()
            
            loss_policy_v = -torch.min(surr1, surr2) + nn.MSELoss()(state_action_val.squeeze(-1), rewards_v) 
            
            loss = loss_entropy + loss_policy_v 

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.net_old.load_state_dict(self.net.state_dict())
        self.buffer.buffer.clear()
            
    def logprob(self, mu, std, action):
        a = ((mu.clamp(-1,1)-action)**2)/(2*std.clamp(min=0.1))
        b = torch.log(torch.sqrt(2*np.pi*std.clamp(min=0.1)))
        return a-b
        
        