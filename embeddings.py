
"""

behavioral embeddings.

This could include all embeddings/a framework to learn those.

"""
import numpy as np

#from dppy.finite_dpps import FiniteDPP

def embed(params, data, policy, states, k=100):

    if params['embedding'] == 'a_s':
        embedding = np.concatenate([policy.forward(x, eval=False) for x in states], axis=0)
    
    return(embedding)
    
    

    
    
    