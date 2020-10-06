# Diversity via Determinants (DvD)

This code is a clean version of the DvD algorithm used in the paper Effective Diversity in Population Based Reinforcement Learning. This only supports the single mode experiments, point, Swimmer, Walker2d, BipedalWalker and HalfCheetah.

In requirements.txt we have all the requirements to run this. There are quite a few, because they came alongside some of the other packages and we thought it was better to be inclusive. The code requires a MuJoCo license, which can be obtained from https://www.roboti.us/license.html. All full-time students get this for free. This should be easy to run, once the MuJoCo license is working.

The code is loosely based on ARS, from here. We have a Learner class which contains the agents, and then each agent induces its own workers to run the rollouts.

To run the `Swimmer` experiments type the following:

`python train.py --env_name Swimmer-v2 --num_workers 4`

The current implementation uses the default settings from the experiments in the paper.


Iter: 1, Eps: 1000, Mean: -932.038, Max: -791.4992, Best: 2, MeanD: 0.6747, MinD: 0.2805, Lam: 0.5
Iter: 2, Eps: 2000, Mean: -886.4041, Max: -847.5858, Best: 2, MeanD: 0.779, MinD: 0.384, Lam: 0
Iter: 3, Eps: 3000, Mean: -845.9665, Max: -829.3266, Best: 0, MeanD: 0.8003, MinD: 0.2165, Lam: 0.5
Iter: 4, Eps: 4000, Mean: -852.7553, Max: -829.0697, Best: 0, MeanD: 0.8917, MinD: 0.3303, Lam: 0
Iter: 5, Eps: 5000, Mean: -820.323, Max: -812.4151, Best: 3, MeanD: 1.3146, MinD: 0.6268, Lam: 0.5
Iter: 6, Eps: 6000, Mean: -817.7337, Max: -810.4426, Best: 3, MeanD: 1.3365, MinD: 0.225, Lam: 0
Iter: 7, Eps: 7000, Mean: -811.7265, Max: -803.4928, Best: 3, MeanD: 1.5173, MinD: 0.2084, Lam: 0
Iter: 8, Eps: 8000, Mean: -802.805, Max: -796.6485, Best: 4, MeanD: 1.29, MinD: 0.476, Lam: 0
Iter: 9, Eps: 9000, Mean: -801.6264, Max: -792.3969, Best: 4, MeanD: 1.4952, MinD: 0.3147, Lam: 0.5
Iter: 10, Eps: 10000, Mean: -814.2167, Max: -800.0378, Best: 0, MeanD: 1.6832, MinD: 0.504, Lam: 0
Iter: 11, Eps: 11000, Mean: -804.1007, Max: -791.8395, Best: 4, MeanD: 1.6615, MinD: 0.6748, Lam: 0
Iter: 12, Eps: 12000, Mean: -799.0066, Max: -779.3149, Best: 4, MeanD: 1.723, MinD: 0.622, Lam: 0
Iter: 13, Eps: 13000, Mean: -776.7099, Max: -719.913, Best: 4, MeanD: 1.9942, MinD: 0.582, Lam: 0
Iter: 14, Eps: 14000, Mean: -734.7749, Max: -670.1611, Best: 4, MeanD: 2.1116, MinD: 1.1812, Lam: 0
Iter: 15, Eps: 15000, Mean: -720.3486, Max: -656.7649, Best: 4, MeanD: 1.602, MinD: 0.7493, Lam: 0
Iter: 16, Eps: 16000, Mean: -686.2283, Max: -638.3012, Best: 3, MeanD: 1.8428, MinD: 0.8319, Lam: 0
Iter: 17, Eps: 17000, Mean: -672.0172, Max: -637.6803, Best: 2, MeanD: 2.5318, MinD: 0.5818, Lam: 0
Iter: 18, Eps: 18000, Mean: -672.0613, Max: -632.3368, Best: 3, MeanD: 2.8482, MinD: 0.2657, Lam: 0
Iter: 19, Eps: 19000, Mean: -659.7307, Max: -625.5453, Best: 2, MeanD: 1.917, MinD: 0.3274, Lam: 0
Iter: 20, Eps: 20000, Mean: -657.1595, Max: -607.0382, Best: 2, MeanD: 3.3323, MinD: 0.3485, Lam: 0