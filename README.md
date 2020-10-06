# Diversity via Determinants (DvD)

This code is a clean version of the DvD-ES algorithm used in the paper Effective Diversity in Population Based Reinforcement Learning, appearing at NeurIPS 2020. 

This only supports the single mode experiments, point, Swimmer, Walker2d, BipedalWalker and HalfCheetah, but you can most certainly add your own!

The main requirements are: ray, gym, numpy, scipy, scikit-learn and pandas. The code requires a MuJoCo license, which can be obtained from [here](https://www.roboti.us/license.html). All full-time students get this for free. This should be easy to run, once the MuJoCo license is working.

The code is loosely based on ARS, from [here](https://github.com/modestyachts/ARS). We have a Learner class which contains the agents, and then each agent induces its own workers to run the rollouts. For the environments used in the paper, we have a file experiments.py which includes the hyperparameters. If you want to try others, then pick the config from the env with the most similar state and action dimensions. 

To run the `Swimmer` experiments type the following:

`python train.py --env_name Swimmer-v2 --num_workers N`

Where N is the number of cores on your machine. Or maybe one less, so you can still do other things :) 

When it runs, you should see something like the following:

```

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
```

This means it has done 20 iterations, 20k episodes, the mean population reward is -657, max is -607. In this case we have a good outcome, because on this environment the local maximum is -780, so we have avoided it! The mean Euclidean distance between the policies is 3.33 and the minimum (between any two) distance is 0.3485 (we want this to be large to reduce redundancy). Lam is the lambda coefficient which in this case adapts between 0 and 0.5. The config for the bandit controller is in the bandits.py file. 

The key parameter to switch between adaptive and a fixed lambda is w_nov (weight for novelty). If you set it to -1 it uses adaptive, otherwise, it uses whatever weight you give it. 

Finally, please do get in touch for further questions, or for help using DvD for other RL algorithms. My email is jackph - at - robots.ox.ac.uk :)

#### Citation

```
@incollection{parkerholder2020effective,
  author    = {Jack Parker{-}Holder and
               Aldo Pacchiano and
               Krzysztof Choromanski and
               Stephen Roberts},
  title     = {Effective {D}iversity in {P}opulation{-}{B}ased {R}einforcement {L}earning},
  year      = {2020},
  booktitle = {Advances in Neural Information Processing Systems 34},
}

```
