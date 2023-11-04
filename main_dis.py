import gym
from collections import deque, namedtuple
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.config import CFG
from agent.buffer import ExperienceDis, ExperienceBufferDis
from agent.agent import NetRnn, NetConv, AgentDis, Net
from environment.env import SpinEnvDis
from ham.ham import random_hamiltonian, random_ferro_hamiltonian

import time


cfg = CFG()

size = 16
reps = 1
steps = 1e4

if __name__ == "__main__":
    
    hamiltonian = random_ferro_hamiltonian(size)
    #print(hamiltonian[0])
    print(hamiltonian)
    env = SpinEnvDis(hamiltonian, size=size, steps = steps)
    print(env.observation_space.shape)
    buffer = ExperienceBufferDis(cfg.REPLAY_SIZE)

    agent = AgentDis(env, buffer, recurrent=cfg.rec)
    energy_mem = list()
    i = 0
    for game in range(cfg.N_GAMES+1):
        env = SpinEnvDis(hamiltonian, size=size, steps=steps)
        done = False
        observation = env.reset()
        game_actions = [] 
        game_obs = list()
        game_next=list()
        game_info=list()
        start_time = time.time()
        r = []
        game_d=list()
        print('start', game)
        while (not done):
            action = agent.choose(observation)

            next_observation, reward, done, _, info = env.step(action)
            r.append(reward)
            game_obs.append(observation)
            game_next.append(next_observation)
            game_info.append(info)
            game_actions.append(action)
            game_d.append(done)
            observation = next_observation
            if observation.sum() == size : 
                print('Yes', observation)
                for i in range(len(r)):
                    agent.step(game_obs[i], game_actions[i], r[i],game_d[i], game_next[i],game_info[i])
                done = True
            if done: 
                energy_mem.append(reward)
                print("done", time.time()-start_time, reward, env.steps)
                env.close()

        if (game % cfg.SYNC == 0) & (game!=0):
            i+=1
            start_time = time.time()
            try : 
                agent.learn()
                print('Sync', time.time()-start_time, reward)
            except : 
                print('No learning')
            #torch.save(agent.net.state_dict(), f'models/{size}-{reps}-{i}.pt')
            
            print(observation)
    
    plt.figure()
    plt.plot(energy_mem)
    plt.show()
    
    print('TEST')
    fer = random_ferro_hamiltonian(size)
    env_test = SpinEnvDis(fer, size=size, steps=steps)
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
            print("done", time.time()-start_time, reward, env_test.steps)
            env_test.close()
            print(observation)