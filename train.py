
from policies import FullyConnected
from optimizers import Adam
from learner import Learner
from utils import *
from shared_noise import *
from es import population_update
from experiments import get_experiment
import simpleenvs
from bandits import BayesianBandits
from embeddings import embed

import psutil
import ray
import gym
import parser
import argparse
import numpy as np
import pandas as pd

import os
import time
from copy import deepcopy
from random import sample 
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from dppy.finite_dpps import FiniteDPP


def select_states(master, params, states):

    # select random states... if you dont have enough just take the ones you have
    if int(params['states'].split('-')[1]) < len(states):
        selected = sample(states, int(params['states'].split('-')[1]))
        return(selected)
    else:
        return(states)


# needed because sometimes the plasma store fills up... ray issue don't FULLY understand :)
def reset_ray(master, params):
    ray.disconnect()
    ray.shutdown()
    time.sleep(5)
    del os.environ['RAY_USE_NEW_GCS']
    ray.init(
        plasma_directory="/tmp")
    os.environ['RAY_USE_NEW_GCS'] = 'True'
    flush_policy = ray.experimental.SimpleGcsFlushPolicy(flush_period_secs=0.1)        
    ray.experimental.set_flushing_policy(flush_policy)
                
def train(params):
    
    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    master = Learner(params)
        
    n_eps = 0
    n_iter = 0
    ts_cumulative = 0
    ts, rollouts, rewards, max_rwds, dists, min_dists, agents, lambdas = [0], [0], [], [], [], [], [], []
    params['num_sensings'] = params['sensings']

    master.agent=0
    # get initial states so you can get behavioral embeddings
    population = [master.agents[x].rollout(env, params['steps'], incl_data=True) for x in master.agents.keys()]
    all_states =  [s[0] for x in population for s in x[1]]
    master.selected = select_states(master, params, all_states)
    master.update_embeddings(params, population)
    master.calc_pairwise_dists(params)
    master.select_agent()
    master.get_agent()
    #initial reward
    reward = master.policy.rollout(env, params['steps'], incl_data=False)

    rewards.append(reward)
    agents.append(master.agent)
    dists.append(master.dists)
    max_reward=reward
    max_rwds.append(max_reward)
    min_dists.append(master.min_dist)
    
    if params['w_nov'] < 0:
        bb = BayesianBandits()
        params['w_nov'] = 0
    lambdas.append(params['w_nov'])
        
    while n_iter < params['max_iter']:

        print('Iter: %s, Eps: %s, Mean: %s, Max: %s, Best: %s, MeanD: %s, MinD: %s, Lam: %s' %(n_iter, n_eps, np.round(reward,4), np.round(max_reward,4), master.agent, np.round(master.dists,4), np.round(master.min_dist,4), params['w_nov']))
       
        if (n_iter>0) & (params['num_agents'] > 1):
            master.calc_pairwise_dists(params)
            master.select_agent()
            master.get_agent()

        ## Main Function Call
        params['n_iter'] = n_iter
        if params['num_agents'] > 1:
            gradient, timesteps = population_update(master, params)
            n_eps += 2*params['num_sensings'] * params['num_agents']
        else:
            gradient, timesteps = individual_update(master, params)
            n_eps += 2*params['num_sensings']
            
        ts_cumulative += timesteps
        all_states += master.states
        if params['num_sensings'] < len(all_states):
            all_states = sample(all_states, params['num_sensings'])

        gradient /= (np.linalg.norm(gradient) / master.policy.N + 1e-8)
                
        n_iter += 1
        update = Adam(gradient, master, params['learning_rate'], n_iter)

        rwds, trajectories = [], []
        if params['num_evals'] > 0:
            seeds = [int(np.random.uniform()*10000) for _ in range(params['num_evals'])]
        for i in range(params['num_agents']):
            master.agent = i
            master.get_agent()
            master.policy.update(master.policy.params + update[(i*master.policy.N):((i+1)*master.policy.N)])
            if params['num_evals'] > 0:
                reward = 0
                for j in range(params['num_evals']):
                    r, traj = master.policy.rollout(env, params['steps'], incl_data=True, seed =seeds[j])
                    reward += r
                reward /= params['num_evals']
            else:
                reward, traj = master.policy.rollout(env, params['steps'], incl_data=True)
            rwds.append(reward)
            trajectories.append(traj)
            if reward > master.best[i]:
                master.best[i] = reward
                np.save('data/%s/weights/Seed%s_Agent%s' %(params['dir'], params['seed'], i), master.policy.params)
            master.reward[i].append(reward)
            master.update_agent()
        reward = np.mean(rwds)
        max_reward = max(rwds)
        traj = trajectories[np.argmax(rwds)]
        master.agent = np.argmax(rwds)

        
        # Update selected states
        master.selected = select_states(master, params, all_states)
        master.update_embeddings(params)
        
        master.embedding = embed(params, traj, master.policy, master.selected)
        rewards.append(reward)
        max_rwds.append(max_reward)
        master.reward[master.agent].append(reward)
        if reward > master.best[master.agent]:
            master.best[master.agent] = reward
            np.save('data/%s/weights/Seed%s_Agent%s' %(params['dir'], params['seed'], master.agent), master.policy.params)
        
        ## update the bandits
        try:
            bb.update_dists(reward)
            params['w_nov'] = bb.sample()
        except NameError:
            pass
        
        lambdas.append(params['w_nov'])
        rollouts.append(n_eps)
        agents.append(master.agent)
        dists.append(master.dists)
        min_dists.append(master.min_dist)
        ts.append(ts_cumulative)
        master.update_agent()

        if n_iter % params['flush'] == 0:
            reset_ray(master, params)
            master.init_workers(params)
        
        out = pd.DataFrame({'Rollouts': rollouts, 'Reward': rewards, 'Max': max_rwds, 'Timesteps': ts, 'Dists': dists, 'Min_Dist':min_dists, 'Agent': agents, 'Lambda': lambdas})
        out.to_csv('data/%s/results/Seed%s.csv' %(params['dir'], params['seed']), index=False) 

def main():
    
    parser = argparse.ArgumentParser()

    ## Env setup
    parser.add_argument('--env_name', type=str, default='point-v0')
    parser.add_argument('--num_agents', '-na', type=int, default=5)
    parser.add_argument('--seed', '-sd', type=int, default=0)
    parser.add_argument('--max_iter', '-it', type=int, default=2000)
    parser.add_argument('--policy', '-po', type=str, default='FC')
    parser.add_argument('--embedding', '-em', type=str, default='a_s')
    parser.add_argument('--num_workers', '-nw', type=int, default=4)
    parser.add_argument('--filename', '-f', type=str, default='')
    parser.add_argument('--num_evals', '-ne', type=int, default=0)
    parser.add_argument('--flush', '-fl', type=int, default=1000) # may need this. it resets ray, because sometimes it fills the memory.
    parser.add_argument('--ob_filter', '-ob', type=str, default='MeanStdFilter') # 'NoFilter'
    parser.add_argument('--w_nov', '-wn', type=float, default=-1) # if negative it uses the adaptive method, else it will be fixed at the value you pick (0,1). Note that if you pick 1 itll be unsupervised (ie no reward)
    parser.add_argument('--dpp_kernel', '-ke', type=str, default='rbf')
    parser.add_argument('--states', '-ss', type=str, default='random-20') # 'random-X' X is how many
    parser.add_argument('--update_states', '-us', type=int, default=20) # how often to update.. we only used 20

    args = parser.parse_args()
    params = vars(args)

    params = get_experiment(params)

    ray.init(plasma_directory="/tmp/")
    os.environ['RAY_USE_NEW_GCS'] = 'True'

    state_word = [str(params['states'].split('-')[0]) if params['w_nov'] > 0 else ''][0]
    params['dir'] = params['env_name'] + '_Net' + str(params['layers']) + 'x' + str(params['h_dim']) + '_Agents' + str(params['num_agents']) + '_Novelty' + str(params['w_nov']) + state_word + 'kernel_' + params['dpp_kernel'] + '_lr' + str(params['learning_rate']) + '_' + params['filename'] + params['ob_filter']
    
    if not(os.path.exists('data/'+params['dir'])):
        os.makedirs('data/'+params['dir'])
        os.makedirs('data/'+params['dir']+'/weights')
        os.makedirs('data/'+params['dir']+'/results')
    
    train(params)

if __name__ == '__main__':
    main()
