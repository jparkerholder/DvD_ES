import numpy as np

def rbf_kernel(x, y, sigma):
	return np.exp(-np.linalg.norm(x-y)**2 / (2 * sigma**2))

def rbf_kernel_grad(x, y, sigma):
	# grad w.r.t. y
	return (x - y) / (sigma**2) * rbf_kernel(x, y, sigma)