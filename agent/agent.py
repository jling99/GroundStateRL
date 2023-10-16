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
        
        self.conv_spin = nn.Conv1d(input_shape[1], cfg.CONV_SPIN_KERNEL)
        self.conv_rep = nn.Conv1d(input_shape[2], cfg.CONV_REP_KERNEL)
        
        self.dense = nn.Linear(cfg.CONV_REP_KERNEL*input_shape[1] + cfg.CONV_SPIN_KERNEL*input_shape[2], cfg.DENSE_HIDDEN)
        
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
        
    def step(self, state, action, reward, done, next_state, info):
            
        exp = Experience(state, action, reward, done, next_state, info)
        self.buffer.append(exp)
        
    def choose(self, state):
        state_action_mu, state_action_std,_ = self.net(torch.tensor(state, dtype=torch.float32))
        action = np.random.normal(state_action_mu.detach().numpy()[0,0], state_action_std.detach().numpy()[0,0],)
        return action

    def learn(self):

        states, actions, rewards, dones = self.buffer.sample(self.net)
        states_v = torch.tensor(states, dtype=torch.float32)
        actions_v = torch.tensor(actions, dtype=torch.float32)
        rewards_v = torch.tensor(rewards, dtype=torch.float32)
        
        state_action_mu, state_action_std, state_action_val = self.net(states_v)
        
        advantage = (-state_action_val[dones].squeeze(-1)+rewards_v[dones]).unsqueeze(-1)
        
        loss_value_v = F.mse_loss(state_action_val.squeeze(-1), rewards_v) #or l1_loss
        self.optimizer.zero_grad()
        
        