

import numpy as np
import gym
import ray


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

        
def evaluate(env, params, p):
    return(p.rollout(env, params['steps'], incl_data=True))

## Adam optimizer


def Adam(dx, learner, learning_rate, t, eps = 1e-8, beta1 = 0.9, beta2 = 0.999):
    learner.m = beta1 * learner.m + (1 - beta1) * dx
    mt = learner.m / (1 - beta1 ** t)
    learner.v = beta2 * learner.v + (1-beta2) * (dx **2)
    vt = learner.v / (1 - beta2 ** t)
    update = learning_rate * mt / (np.sqrt(vt) + eps)
    return(update)

## Shared noise table

@ray.remote
def create_shared_noise():
    """
    Create a large array of noise to be shared by all workers. Used 
    for avoiding the communication of the random perturbations delta.
    """

    seed = 12345
    count = 2500000
    noise = np.random.RandomState(seed).randn(count).astype(np.float64)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise, seed = 11):

        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == np.float64

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)

## Kernels

def rbf_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * sigma**2))

def rbf_kernel_grad(x, y, sigma):
    # grad w.r.t. y
    return (x - y) / (sigma**2) * rbf_kernel(x, y, sigma)


## Filter 


class Filter(object):
    """Processes input, possibly statefully."""

    def update(self, other, *args, **kwargs):
        """Updates self with "new state" from other filter."""
        raise NotImplementedError

    def copy(self):
        """Creates a new object with same state as self.
        Returns:
            copy (Filter): Copy of self"""
        raise NotImplementedError

    def sync(self, other):
        """Copies all state from other filter to self."""
        raise NotImplementedError


class NoFilter(Filter):
    def __init__(self, *args):
        pass

    def __call__(self, x, update=True):
        return np.asarray(x, dtype = np.float64)

    def update(self, other, *args, **kwargs):
        pass

    def copy(self):
        return self

    def sync(self, other):
        pass

    def stats_increment(self):
        pass

    def clear_buffer(self):
        pass

    def get_stats(self):
        return 0, 1

    @property
    def mean(self):
        return 0

    @property
    def var(self):
        return 1

    @property
    def std(self):
        return 1



# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):

    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype = np.float64)
        self._S = np.zeros(shape,  dtype = np.float64)
        self._M2 = np.zeros(shape,  dtype = np.float64)

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M = np.copy(self._M)
        other._S = np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        # Unvectorized update of the running statistics.
        assert x.shape == self._M.shape, ("x.shape = {}, self.shape = {}"
                                          .format(x.shape, self._M.shape))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            deltaM2 = np.square(x) - self._M2
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n
            

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(
            self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class MeanStdFilter(Filter):
    """Keeps track of a running mean for seen states"""

    def __init__(self, shape, demean=True, destd=True):
        self.shape = shape
        self.demean = demean
        self.destd = destd
        self.rs = RunningStat(shape)
        # In distributed rollouts, each worker sees different states.
        # The buffer is used to keep track of deltas amongst all the
        # observation filters.

        self.buffer = RunningStat(shape)

        self.mean = np.zeros(shape, dtype = np.float64)
        self.std = np.ones(shape, dtype = np.float64)

    def clear_buffer(self):
        self.buffer = RunningStat(self.shape)
        return

    def update(self, other, copy_buffer=False):
        """Takes another filter and only applies the information from the
        buffer.
        Using notation `F(state, buffer)`
        Given `Filter1(x1, y1)` and `Filter2(x2, yt)`,
        `update` modifies `Filter1` to `Filter1(x1 + yt, y1)`
        If `copy_buffer`, then `Filter1` is modified to
        `Filter1(x1 + yt, yt)`.
        """
        self.rs.update(other.buffer)
        if copy_buffer:
            self.buffer = other.buffer.copy()
        return 

    def copy(self):
        """Returns a copy of Filter."""
        other = MeanStdFilter(self.shape)
        other.demean = self.demean
        other.destd = self.destd
        other.rs = self.rs.copy()
        other.buffer = self.buffer.copy()
        return other

    def sync(self, other):
        """Syncs all fields together from other filter.
        Using notation `F(state, buffer)`
        Given `Filter1(x1, y1)` and `Filter2(x2, yt)`,
        `sync` modifies `Filter1` to `Filter1(x2, yt)`
        """
        assert other.shape == self.shape, "Shapes don't match!"
        self.demean = other.demean
        self.destd = other.destd
        self.rs = other.rs.copy()
        self.buffer = other.buffer.copy()
        return

    def __call__(self, x, update=True):
        x = np.asarray(x, dtype = np.float64)
        if update:
            if len(x.shape) == len(self.rs.shape) + 1:
                # The vectorized case.
                for i in range(x.shape[0]):
                    self.rs.push(x[i])
                    self.buffer.push(x[i])
            else:
                # The unvectorized case.
                self.rs.push(x)
                self.buffer.push(x)
        if self.demean:
            x = x - self.mean
        if self.destd:
            x = x / (self.std + 1e-8)
        return x

    def stats_increment(self):
        self.mean = self.rs.mean
        self.std = self.rs.std

        # Set values for std less than 1e-7 to +inf to avoid 
        # dividing by zero. State elements with zero variance
        # are set to zero as a result. 
        self.std[self.std < 1e-7] = float("inf") 
        return

    def get_stats(self):
        return self.rs.mean, (self.rs.std + 1e-8)

    def __repr__(self):
        return 'MeanStdFilter({}, {}, {}, {}, {}, {})'.format(
            self.shape, self.demean,
            self.rs, self.buffer)

    
def get_filter(filter_config, shape = None):
    if filter_config == "MeanStdFilter":
        return MeanStdFilter(shape)
    elif filter_config == "NoFilter":
        return NoFilter()
    else:
        raise Exception("Unknown observation_filter: " +
                        str(filter_config))



