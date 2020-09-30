
import numpy as np

def Adam(dx, learner, learning_rate, t, eps = 1e-8, beta1 = 0.9, beta2 = 0.999):
    learner.m = beta1 * learner.m + (1 - beta1) * dx
    mt = learner.m / (1 - beta1 ** t)
    learner.v = beta2 * learner.v + (1-beta2) * (dx **2)
    vt = learner.v / (1 - beta2 ** t)
    update = learning_rate * mt / (np.sqrt(vt) + eps)
    return(update)