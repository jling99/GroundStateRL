import gym
from gym import spaces
import numpy as np



class SpinEnv(gym.Env):
    
    def __init__(self, ham, size=16, reps=64, sweeps=10, steps = 100):
        self.size = size
        self.reps = reps
        self.sweeps = sweeps
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.reps, self.size, 1))
        self.links = ham #dict of (neighb, couple)
        self.stop = steps

        
    def _get_obs(self):
        return self.lattice  
        
    def reset(self):
        self.lattice = self.init_lattice()
        self.beta = self.init_beta()
        self.site_energy = self.calc_energy()
        self.step = 0
        pass
    
    def step(self, action):
        
        if self.beta[0] + action < 1e-5 : 
            action = 0
            
        for i in range (self.reps):
            self.beta[i] =+ action/self.sweeps
            self.sweep(i)
            
        self.step +=1
                
        observation = self._get_obs()
        reward = 0
        done = False
                
        if self.step >= self.stop:
            done = True
            reward = self.site_energy.sum(axis=1).max()
        
        return observation, reward, done, False, {}                
    
    def flip(self, rep, site):
        self.lattice[rep, site] *=-1
        self.site_energy = self.calc_energy()

    def sweep(self, rep):
        for i in range(self.sweeps):
            j = np.random.randint(0,self.size)
            
            if self.site_energy[rep, j]< -np.log(np.random.uniform(0,1))/(2*self.beta):
                self.flip(rep,j)

    def init_lattice(self):
        return np.random.choice([-1,1],(self.reps, self.size))
    
    def init_beta(self):
        self.beta = np.ones(self.reps)
    
    def calc_energy(self):
        
        all_en = np.zeros((self.reps, self.size))
        for i in range(self.reps):
            for k,v in self.links.items():
                en = 0
                if len(v)>0:
                    for c in v :
                        if k == c[0]:
                            en += self.lattice[i, k]*c[1]
                        else : 
                            en+= self.lattice[i, k]*c[1]*self.lattice[i, c[0]]
                all_en[i,k] = -en
        
        return all_en