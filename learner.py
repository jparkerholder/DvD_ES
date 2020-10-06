
## Requirements
from utils import *
from policies import get_policy
import simpleenvs
from worker import Worker
from embeddings import embed

## External
import ray
import gym
#import mujoco_py
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import rankdata
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from dppy.finite_dpps import FiniteDPP

class Learner(object):

    def __init__(self, params):

        params['zeros'] = False
        self.agents = {i:get_policy(params, params['seed']+1000*i) for i in range(params['num_agents'])}

        self.timesteps = 0

        self.w_reward = 1
        self.w_size = 0
        self.dists = 0

        self.adam_params = {i:[0,0] for i in range(params['num_agents'])}

        self.buffer = []
        self.states = []
        self.embeddings = {i:[] for i in range(params['num_agents'])}
        self.best = {i:-9999 for i in range(params['num_agents'])}
        self.reward = {i:[-9999] for i in range(params['num_agents'])}
        self.min_dist = 0
        
        self.num_workers = params['num_workers']
        self.init_workers(params)
        
    def init_workers(self, params):
        
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = params['seed'] + 3)
        
        self.workers = [Worker.remote(params['seed'] + 7 * i,
                                      env_name=params['env_name'],
                                      policy=params['policy'],
                                      h_dim = params['h_dim'],
                                      layers = params['layers'],
                                      deltas= deltas_id,
                                      rollout_length=params['steps'],
                                      delta_std=params['sigma'],
                                      num_evals=params['num_evals'],
                                      ob_filter=params['ob_filter']) for i in range(params['num_workers'])]


    def get_agent(self):
        self.policy = deepcopy(self.agents[self.agent])
        self.embedding = self.embeddings[self.agent].copy()
        self.m = self.adam_params[self.agent][0]
        self.v = self.adam_params[self.agent][1]

    def update_agent(self):
        self.agents[self.agent] = deepcopy(self.policy)
        self.embeddings[self.agent] = self.embedding.copy()
        self.adam_params[self.agent] = [self.m, self.v]
    
    def update_embeddings(self, params, data=[]):
        
        for j in range(params['num_agents']):
            if params['embedding'] == 'a_s':
                self.embeddings[j] = [embed(params, [], self.agents[j], self.selected)]
            else:
                self.embeddings[j] = [embed(params, s, self.agents[j], self.selected) for s in data[j][1]]

    def calc_pairwise_dists(self, params):
        
        dists = np.zeros([params['num_agents'], params['num_agents']])
        
        min_dist = 999
        for i in range(params['num_agents']):
            for j in range(params['num_agents']):
                dists[i][j] = np.linalg.norm(self.embeddings[i][0] - self.embeddings[j][0])
                if (i != j) & (dists[i][j] < min_dist):
                    min_dist = dists[i][j] 
    
        self.dists = np.mean(dists)
        self.min_dist = min_dist
        self.dist_vec = np.mean(dists, axis=1)
        self.dist_vec /= np.sum(self.dist_vec)
        
    def select_agent(self):
        if min([x[-1] for x in list(self.reward.values())]) > -9999:
            reward_vec = rankdata([max(x[-5:]) for x in list(self.reward.values())])
            reward_vec /= np.sum(reward_vec)
            
            dist_vec = rankdata(self.dist_vec)
            dist_vec /= np.sum(dist_vec)
            vec = (dist_vec + reward_vec)/2
            self.agent = np.argmax(np.random.multinomial(1, vec))
        else:
            self.agent = np.argmax(np.random.multinomial(1, self.dist_vec))

    
