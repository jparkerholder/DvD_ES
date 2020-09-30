# Diversity via Determinants (DvD)

This code is a clean version of the DvD algorithm used in the paper Effective Diversity in Population Based Reinforcement Learning. This only supports the single mode experiments, point, Swimmer, Walker2d, BipedalWalker and HalfCheetah.

In requirements.txt we have all the requirements to run this. There are quite a few, because they came alongside some of the other packages and we thought it was better to be inclusive. The code requires a MuJoCo license, which can be obtained from https://www.roboti.us/license.html. All full-time students get this for free. This should be easy to run, once the MuJoCo license is working.

The code is loosely based on ARS, which is now integrated into rllib. We have a Learner class which contains the agents, and then each agent induces its own workers to run the rollouts.

To run the `Swimmer` experiments type the following:

`python train.py --env_name Swimmer-v2 --num_workers 4`

The current implementation uses the default settings from the experiments in the paper.
