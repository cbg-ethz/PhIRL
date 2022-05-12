"""Submodule implementing the MaxEnt IRL
approach to finding the reward function.

The MDP is deterministic and each state
is a set of already acquired mutations.
We will encode each state as a binary tuple
(0, 1, 0, ..., 0),
and 1 on ith position means that the mutation
i+1 has been acquired.

MaxEnt IRL was proposed in
B.D. Ziebart et al., Maximum Entropy Inverse Reinforcement Learning,
AAAI (2008), https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf
"""
# Note: This file should be empty. We put the public interface into api.py
