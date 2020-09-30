import ray
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
import pandas as pd
import time
import os
from utils import *

def normalize(data, wrt):
    data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-8)
    return np.clip(data, -2, 2)

def normalize2(data, wrt):
    # data = data to normalize
    # wrt = data will be normalized with respect to this
    return (data - np.min(wrt, axis=0))/(np.max(wrt,axis=0) - np.min(wrt,axis=0))

def get_det(pop, params):

    if params['dpp_kernel'] == 'Matern5':
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=2.5)
        K = kernel(pop)  
    elif params['dpp_kernel'] == 'Matern3':
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)
        K = kernel(pop)
    elif params['dpp_kernel'] == 'Exponential':
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=0.5) #matern 1/2 = exponential kernel
        K = kernel(pop)
    elif params['dpp_kernel'] == 'RQ':
        kernel = RationalQuadratic()
        K = kernel(pop)
    elif params['dpp_kernel'] == 'Linear':
        pop = normalize2(pop, pop)
        row_sums = pop.sum(axis=1)
        pop = pop / (row_sums[:, np.newaxis] + 1e-8)
        K = linear_kernel(pop) 
    else:
        K = rbf_kernel(pop)
        
    return(np.linalg.det(K))


def population_update(master, params):

    timesteps = 0  
    rwds, embeddings, agent_deltas, data = [], [], [], []
    num_rollouts = int(params['num_sensings'] / params['num_workers'])
    params['num_sensings'] = int(num_rollouts * params['num_workers'])
    
    # get rewards/trajectory info
    for i in range(params['num_agents']):
    
        filter_id = ray.put(master.agents[i].observation_filter)
        setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in master.workers]
        ray.get(setting_filters_ids)
        increment_filters_ids = [worker.stats_increment.remote() for worker in master.workers]
        ray.get(increment_filters_ids)
        
        use_states = [1 if params['embedding'] == 'a_s' else 0][0]
        policy_id = ray.put(master.agents[i].params)
        rollout_ids = [worker.do_rollouts.remote(policy_id, num_rollouts, master.selected, use_states) for worker in master.workers]
        results = ray.get(rollout_ids)
        
        for j in range(params['num_workers']):
            master.agents[i].observation_filter.update(ray.get(master.workers[j].get_filter.remote()))
        master.agents[i].observation_filter.stats_increment()
        master.agents[i].observation_filter.clear_buffer()
    
        # harvest the worker data.. quite a lot of stuff
        rollout_rewards, deltas_idx, sparsities, emb_selected = [], [], [], []
        for result in results:
            deltas_idx += result['deltas_idx']
            rollout_rewards  += result['rollout_rewards']
            timesteps += result['steps']
            sparsities += result['sparsities']
            data += result['data']
            emb_selected += result['embedding']
        
        rwds.append(np.array(rollout_rewards))
        embeddings.append(emb_selected)
        agent_deltas.append(np.array(deltas_idx))

    # Get the correspinding determinants
    if params['w_nov'] > 0:
        dets = np.zeros(np.array(rollout_rewards).shape)
        for i in range(num_rollouts*params['num_workers']):
            pop = np.concatenate(([x[i][0].reshape(embeddings[0][0][0].size,1) for x in embeddings]), axis=1).T
            pop = normalize(pop, pop)
            dets[i, 0] = get_det(pop, params)
                
            pop = np.concatenate(([x[i][1].reshape(embeddings[0][0][0].size,1) for x in embeddings]), axis=1).T
            pop = normalize(pop, pop)        
            dets[i, 1] = get_det(pop, params)

        dets = (dets - np.mean(dets))/(np.std(dets) + 1e-8)
    else:
        dets = np.zeros(np.array(rollout_rewards).shape)   
    
    # pass all the aggregate info to the master policy        
    master.buffer = data
    
    # add a random sample of the states to a state buffer, then only keep last 10 iterations
    master.states = [x[0] for t in data for x in t[0]] + [x[0] for t in data for x in t[1]]

    # individually update the policies
    g_hat = []
    for i in range(params['num_agents']):
        deltas_idx = np.array(agent_deltas[i])
        rollout_rewards = np.array(rwds[i], dtype = np.float64)
        rollout_rewards = (rollout_rewards - np.mean(rollout_rewards)) / (np.std(rollout_rewards) +1e-8)
        rollout_rewards = params['w_nov'] * dets + (1-params['w_nov']) * rollout_rewards
    
        g, count = batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                      (master.deltas.get(idx, master.policy.params.size)
                                                       for idx in deltas_idx),
                                                      batch_size = 500)
        g /= deltas_idx.size
        g_hat.append(g)
    g_hat = np.concatenate(g_hat)
    
    return(g_hat, timesteps)