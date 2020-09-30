

"""
Details of all the experiments we run. 
We do not seek to tune these parameters too much.
The parameters here work for baselines.
"""

def get_experiment(params):
	if params['env_name'] in ['HalfCheetah-v2','HalfCheetah-v1']:
		params['h_dim'] = 32
		params['layers'] = 2
		params['sensings'] = 100
		params['learning_rate'] = 0.05
		params['sigma'] = 0.1
		params['steps'] = 1000
	elif params['env_name'] in ['Walker2d-v2']:
		params['h_dim'] = 32
		params['layers'] = 2
		params['sensings'] = 100
		params['learning_rate'] = 0.05
		params['sigma'] = 0.1
		params['steps'] = 1000
	elif params['env_name'] == 'Swimmer-v2':
		params['h_dim'] = 16
		params['layers'] = 2
		params['sensings'] = 100
		params['learning_rate'] = 0.05
		params['sigma'] = 0.1
		params['steps'] = 1000
	elif params['env_name'] == 'BipedalWalker-v2':
		params['h_dim'] = 32
		params['layers'] = 2
		params['sensings'] = 100
		params['learning_rate'] = 0.05
		params['sigma'] = 0.1
		params['steps'] = 1600
	elif params['env_name'] == 'point-v0':
		params['h_dim'] = 16
		params['layers'] = 2
		params['sensings'] = 100
		params['learning_rate'] = 0.05
		params['sigma'] = 0.1
		params['steps'] = 50
	return(params)
