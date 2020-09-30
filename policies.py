
from filter import get_filter


import numpy as np
from scipy.special import softmax
            
class FullyConnected(object):
    
    def __init__(self, params, seed=0):
        
        np.random.seed(seed)
        
        self.layers = params['layers']
        self.hidden = {}
        self.bias = {}
        
        self.observation_filter = get_filter(params['ob_filter'], shape = (params['ob_dim'],))
        self.update_filter = True
        
        self.hidden['h1'] = np.random.randn(params['h_dim'], params['ob_dim'])/np.sqrt(params['h_dim']*params['ob_dim'])
        self.bias['b1'] = np.random.randn(params['h_dim'])/np.sqrt(params['h_dim'])
        
        if params['layers'] >1:
            for i in range(2, params['layers']+1):
                self.hidden['h%s' %str(i)] = np.random.randn(params['h_dim'], params['h_dim'])/np.sqrt(params['h_dim']*params['h_dim'])
                self.bias['b%s' %str(i)] = np.random.randn(params['h_dim'])/np.sqrt(params['h_dim'])

        self.hidden['h999'] = np.random.randn(params['ac_dim'], params['h_dim'])/np.sqrt(params['ac_dim']*params['h_dim'])
        
        self.w_hidden = np.concatenate([self.hidden[x].reshape(self.hidden[x].size, ) for x in self.hidden.keys()])
        self.w_bias = np.concatenate([self.bias[x].reshape(self.bias[x].size, ) for x in self.bias.keys()])
        
        self.params = np.concatenate((self.w_hidden, self.w_bias))
        self.used = 1
        
        self.N = self.params.size
        
    def get_observation_filter(self):
        return self.observation_filter
    
    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    
    def forward(self, x, eval=True):
        
        x = self.observation_filter(x, update=self.update_filter)
        
        self.used = 0
        a = x.copy()
        for i in range(1, self.layers+1):
            a = np.tanh(np.dot(self.hidden['h%s' %str(i)], a) + self.bias['b%s' %str(i)])
                   
        action = np.tanh(np.dot(self.hidden['h999'], a))
        return(action)
    
    def update(self, w):
        
        w_hidden = w[:self.w_hidden.size]
        w = w[self.w_hidden.size:]
        w_bias = w
        
        for i in range(1, len(self.hidden.keys())):
            update = w_hidden[:self.hidden['h%s' %i].size]
            w_hidden = w_hidden[self.hidden['h%s' %i].size:]
            self.hidden['h%s' %i] = update.reshape(self.hidden['h%s' %i].shape)  
        self.hidden['h999'] = w_hidden.reshape(self.hidden['h999'].shape)
        
        for i in range(1, len(self.bias.keys())+1):
            update = w_bias[:self.bias['b%s' %i].size]
            w_bias = w_bias[self.bias['b%s' %i].size:]
            self.bias['b%s' %i] = update.reshape(self.bias['b%s' %i].shape)  
        
        self.w_hidden = np.concatenate([self.hidden[x].reshape(self.hidden[x].size, ) for x in self.hidden.keys()])
        self.w_bias = np.concatenate([self.bias[x].reshape(self.bias[x].size, ) for x in self.bias.keys()])
        
        self.params = np.concatenate((self.w_hidden, self.w_bias))
        
    def rollout(self, env, steps, incl_data=False, seed=0, train=True):
        if not hasattr(env, 'tasks'):
            env.seed(seed)
        state = env.reset()
        env._max_episode_steps = steps
        total_reward = 0
        done = False
        data=[]
        while not done:
            action = self.forward(state)
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            action = action.reshape(len(action), )
            state, reward, done, _ = env.step(action)
            total_reward += reward
            data.append([state, reward, action])
        self.observation_filter.stats_increment()
        if incl_data:
            return(total_reward, data)
        else:
            return(total_reward)