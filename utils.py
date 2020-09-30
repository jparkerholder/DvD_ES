
from policies import FullyConnected

import numpy as np
import gym


## Utility funcs courtesy of https://github.com/modestyachts/ARS/blob/master/code/utils.py
# plus some extras..

def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def get_policy(params, seed=None):

    if seed:
        params['seed'] = seed

    return(FullyConnected(params, params['seed']))

        
def evaluate(env, params, p):
    return(p.rollout(env, params['steps'], incl_data=True))

