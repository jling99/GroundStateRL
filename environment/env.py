import gym
from gym import spaces
import numpy as np



class SpinEnv(gym.Env):
    
    def __init__(self, ham, size=16, reps=64, steps = 10000, beta = 1):
        self.size = size
        self.reps = reps
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.reps, self.size, 1))
        self.links = ham[0] #dict of (neighb, couple)
        self.all_links = ham[1]
        self.stop = steps
        self.start_beta = beta

        
    def _get_obs(self):
        return self.lattice  
    
    def _get_info(self):
        return {'beta' : self.beta[0],
                'min_enery' : -self.site_rel_energy.sum(axis=1).min(),
                'step' : self.steps}
        
    def reset(self):
        self.lattice = self.init_lattice()
        self.beta = self.init_beta()
        self.site_rel_energy = self.calc_rel_energy()
        self.site_energy = self.calc_energy()
        self.steps = 0

    def step(self, action):
        
        if self.beta[0] + action < 0 : 
            action = 0
            
        for i in range (self.reps):
            self.beta[i] =+ action
            self.sweep(i)
            
        self.steps +=1
                
        observation = self._get_obs()
        reward = 0
        done = False
        info = self._get_info()
                
        if self.steps >= self.stop:
            done = True
            reward = -self.site_rel_energy.sum(axis=1).min()

        return observation, reward, done, False, info              
    
    def flip(self, rep, site):
        self.lattice[rep, site] *=-1
        self.site_rel_energy = self.calc_rel_energy()
        self.site_energy = self.calc_energy()

    def sweep(self, rep):
        for i in range(self.reps):
            j = np.random.randint(0,self.size)
            
            if self.site_energy[rep, j]< -np.log(np.random.uniform(0,1))/(2*self.beta[j]):
                self.flip(rep,j)

    def init_lattice(self):
        return np.random.choice([-1,1],(self.reps, self.size))
    
    def init_beta(self):
        self.beta = np.ones(self.reps)*self.start_beta
    
    def calc_rel_energy(self):
        
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
    
    def calc_energy(self):
        
        all_en = np.zeros((self.reps, self.size))
        for i in range(self.reps):
            for k,v in self.all_links.items():
                en = 0
                if len(v)>0:
                    for c in v :
                        if k == c[0]:
                            en += self.lattice[i, k]*c[1]
                        else : 
                            en+= self.lattice[i, k]*c[1]*self.lattice[i, c[0]]
                all_en[i,k] = -en
        
        return all_en
    
