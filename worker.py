
## Requirements
from utils import SharedNoiseTable
from policies import get_policy

## External
import ray
import gym
import numpy as np

@ray.remote
class Worker(object):  
    
    import simpleenvs
        
    def __init__(self, env_seed,
                 env_name='',
                 shift=0,
                 policy='FC',
                 h_dim=64,
                 layers=2,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02,
                 num_evals=0,
                 ob_filter='NoFilter'):
        
        self.params = {}
        self.env_name = env_name
        self.params['env_name'] = env_name
        self.env = gym.make(env_name)
        self.params['ob_dim'] = self.env.observation_space.shape[0]
        self.params['ac_dim'] = self.env.action_space.shape[0]
        self.env.seed(0)

        self.params['h_dim'] = h_dim
        self.steps = rollout_length
                
        self.params['zeros'] = True
        self.params['seed'] = 0
        self.params['layers'] = layers
        self.shift = shift
        self.sigma = 1
        self.num_evals = num_evals
        self.params['ob_filter'] = ob_filter
        self.policy = get_policy(self.params)

        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.delta_std = delta_std

    def do_rollouts(self, policy, num_rollouts, selected_states, use_states=0, indices = None, seed=0, train=True):

        rollout_rewards, deltas_idx, sparsities, data, embeddings = [], [], [], [], []

        steps = 0

        for i in range(num_rollouts):
            
            if indices is None:
                idx, delta = self.deltas.get_delta(policy.size)
            else:
                idx = indices[i]
                delta = self.deltas.get(idx, policy.size)
            delta = (self.delta_std * delta).reshape(policy.shape)
            deltas_idx.append(idx)
            
            self.policy.update(policy + delta)
            pos_reward, pos_steps, pos_sparse, pos_data = self.rollouts(seed, train)
            if use_states:
                pos_embedding = np.concatenate([self.policy.forward(x, eval=False) for x in selected_states], axis=0)
            else: 
                pos_embedding = []
            
            self.policy.update(policy - delta)
            neg_reward, neg_steps, neg_sparse, neg_data = self.rollouts(seed, train)
            if use_states:
                neg_embedding = np.concatenate([self.policy.forward(x, eval=False) for x in selected_states], axis=0)
            else:
                neg_embedding = []

            rollout_rewards.append([pos_reward, neg_reward])
            sparsities.append([pos_sparse, neg_sparse])
            data.append([pos_data, neg_data])
            steps += pos_steps + neg_steps
            embeddings.append([pos_embedding, neg_embedding])
        
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, 
        'sparsities': sparsities, 'steps' : steps, 'data': data, 'embedding': embeddings}
    
    def rollouts(self, seed=0, train=True):
        self.env._max_episode_steps = self.steps
        if self.num_evals > 0:
            total_reward = 0
            timesteps = 0
            sparsity = self.policy.used
            data = []
            for _ in range(self.num_evals):
                self.env.seed(None)
                state = self.env.reset()
                reward, ts, sp, d = self.rollout(state)
                sparsity += sp
                total_reward += reward
                timesteps += ts
                data += d
        else:
            if not hasattr(self.env, 'tasks'):
                self.env.seed(seed)
            state = self.env.reset()
            total_reward, timesteps, sparsity, data = self.rollout(state)

        return(total_reward, timesteps, sparsity, data)

    def rollout(self, state):
        total_reward = 0
        done = False
        timesteps = 0
        sparsity = self.policy.used
        data = []
        while not done:
            action = self.policy.forward(state)
            if hasattr(self.env, 'envtype'):
                if self.env.envtype == 'dm':
                    action = np.clip(action, self.env.env.action_spec().minimum, self.env.env.action_spec().maximum)
                else:
                    action = np.clip(action, self.env.env.action_space.low[0], self.env.env.action_space.high[0])
                action = action.reshape(len(action), )
            elif self.env_name.split(':')[0] != 'bsuite':
                action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
                action = action.reshape(len(action), )
            state, reward, done, _ = self.env.step(action)
            total_reward += reward - self.shift
            timesteps += 1
            data.append([state, reward, np.array(action)])
        return(total_reward, timesteps, sparsity, data)
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return