import gym
from collections import deque, namedtuple
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.config import CFG
from agent.buffer import Experience, ExperienceBuffer
from agent.agent import NetRnn, NetConv, Agent
from environment.env import SpinEnv
from ham.ham import random_hamiltonian, random_ferro_hamiltonian

import time


cfg = CFG()

size = 16
reps = 32
steps = 1e3

if __name__ == "__main__":
    
    hamiltonian = random_hamiltonian(size)
    #print(hamiltonian[0])
    env = SpinEnv(hamiltonian, size=size, reps=reps, steps = steps)
    print(env.observation_space.shape)
    buffer = ExperienceBuffer(cfg.REPLAY_SIZE)

    agent = Agent(env, buffer, recurrent=cfg.rec)
    energy_mem = list()
    i = 0
    for game in range(cfg.N_GAMES+1):
        env = SpinEnv(hamiltonian, size=size, reps=reps)
        done = False
        observation = env.reset()
        game_actions = [] 
        start_time = time.time()
        print('start', game)
        while (not done):
            action = agent.choose(observation)
            #action = np.random.normal()/10
            next_observation, reward, done, _, info = env.step(action)

            agent.step(observation, action, reward,done, next_observation,info)
            
            game_actions.append(action)
            observation = next_observation

            if done: 
                energy_mem.append(reward)
                print("done", time.time()-start_time, reward, env.flips)
                env.close()

        if (game % cfg.SYNC == 0) & (game!=0):
            i+=1
            start_time = time.time()
            agent.learn()
            torch.save(agent.net.state_dict(), f'models/{size}-{reps}-{i}.pt')
            print('Sync', time.time()-start_time)
            print(observation)
    
    plt.figure()
    plt.plot(energy_mem)
    plt.show()
    
    print('TEST')
    fer = random_ferro_hamiltonian(size)
    env_test = SpinEnv(fer, size=size, reps=reps, steps=steps)
    done = False
    observation = env_test.reset()
    game_actions = [] 
    start_time = time.time()

    while (not done):
        action = agent.choose(observation)
        next_observation, reward, done, _, info = env_test.step(action)

        agent.step(observation, action, reward,done, next_observation,info)
        
        game_actions.append(action)
        observation = next_observation

        if done: 
            energy_mem.append(reward)
            print("done", time.time()-start_time, reward, env_test.flips)
            env_test.close()