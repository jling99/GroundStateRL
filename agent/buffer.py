import gym
from collections import deque, namedtuple
import numpy as np

from agent.config import CFG

import torch

cfg = CFG()

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state', 'info'])

#Keep steps in memory
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, net):
        indices = np.arange(len(self.buffer))
        states, actions, rewards, dones, next_states, info = zip(*[self.buffer[idx] for idx in indices])
        
        not_done_idx = []
        last_states = []
        
        #Keep only not terminated steps
        for idx, exp in enumerate(next_states):
            if exp is not None:
                not_done_idx.append(idx)
            last_states.append(np.array(exp, copy=False))
        not_done_idx = np.array(not_done_idx)
        rewards = np.array(rewards, dtype=np.float32)
        
        if not_done_idx.shape[0]>0:
            last_states_v = torch.tensor(np.array(last_states), dtype=torch.float32)
            last_vals_v = net(last_states_v)[2]
            last_vals_np = last_vals_v.data.numpy()[:, 0]
            rewards[not_done_idx] += cfg.GAMMA ** cfg.REWARD_STEPS * last_vals_np
               
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(not_done_idx, dtype=np.uint8)
               
    
