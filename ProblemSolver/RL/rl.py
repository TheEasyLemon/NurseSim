'''
Use Reinforcement Learning methods to solve the revenue
management problem for nurses. We solve the dynamic case
but enforce a policy constraint to solve the static case.

--- System Dynamics ---

State: the state of the system is captured by (j, l), where
j is an integer in {1..m}, where m is the number of nurses.
l is an integer in {0..n}, where n is the required nurses
for the shift. j is the nurse that is going to arrive to schedule
and l is the number of spots left that still need to be filled.
The terminal state where all nurses have arrived is (m+1, l)
for all l in {0..n}.

Parameters: We have parameters p_1..p_m, the probability the nurse
selects the shift. We have parameters q_1..q_m, the probability
the nurse shows up to the shift. We also have r_1..r_m, the revenue
of the shift. We define, as shorthand, v_1..v_m equal to the value
v = pqr.

Actions: At each state, we can choose to either schedule or not 
schedule. If we start at (j, l) and we schedule, we move to the state
(j + 1, l - 1) with probability p_j and gain reward v_j. With
probability (1 - p_j), we move to (j + 1, l) and gain no reward.

Dynamical Equation: see dynamics_flowchart.png.

Example: see example.png.

--- Policies ---

A policy is a deterministic function \pi(a | s) that determines
what action to take given a certain state. The dynamic case
allows the agent to observe the full state (j, l), but the 
problem we consider is the static case, where the agent only
sees (j).

--- A Hunch for Static Policies ---

A hunch that I have is that we can find the optimal static
policy from the dynamic one. Because of the "nice" structure
of the MDP, we can calculate the probability of being in 
a state (j, l) given j (that is, P{s=(j, l) | j = j'}). Then,
to find the optimal static policy, we might take the weighted
average of the actions given j = j' with the associated probability
of being in the state and round towards either 0 or 1.

Dawson Ren, 4/12/23
'''
from typing import Tuple, Dict, List

# map state to action
DynamicPolicy = Dict[Tuple[int, int], int]
# optimal action for each index, where the index
# corresponds to j
StaticPolicy = List[int]

import numpy as np

class NurseRL:
    def __init__(self, p: np.ndarray, q: np.ndarray, r: np.ndarray, n: int) -> None:
        if len(p) != len(q) or len(q) != len(r): raise Exception('p, q, r lengths not the same')
        self.m = len(p)
        if n > self.m: raise Exception('Not enough nurses')

        self.p = p
        self.q = q
        self.r = r
        self.n = n

        self.v = p * q * r

        # States - see example.png for more clarity
        self.S = [(j, l) for j in range(1, self.m + 2) for l in range(1, n + 1) if j + l > self.n]
        self.S.append((self.m + 1, 0))
        

    def step(self, s: int, a: int) -> Tuple[Tuple[int, int], int, bool]:
        '''
        Uses the dynamical system function p(s', r | s, a).
        Note that the dynamics_flowchart.png is 1-indexed, while
        Python is 0-indexed.

        Returns:
        - s' - the state tuple [j, l]
        - r - the reward
        - terminated - whether or not we have reached the terminal state
        '''
        j, l = s

        next_j, next_l = j, l
        reward = 0
        terminated = False

        if a == 1:
            if l == 1:
                # the nurse chooses the shift
                if np.random.random() < self.p[j - 1]:
                    next_j, next_l = self.m, 0
                    reward = self.v[j - 1]
                # the nurse doesn't choose the shift
                else:
                    next_j = j + 1
            else:
                # the nurse chooses the shift
                if np.random.random() < self.p[j - 1]:
                    next_j, next_l = j + 1, l - 1
                    reward = self.v[j - 1]
                # the nurse doesn't choose the shift
                else:
                    next_j = j + 1
        else:
            next_j = j + 1

        if next_j == self.m or next_l == 0:
            terminated = True

        return (next_j, next_l), reward, terminated
    
    def iteration(self) -> DynamicPolicy:
        '''
        Perform value iteration on the MDP
        From Sutton, Barto 2017, Chapter 4.4
        Return a dictionary mapping state to deterministic action.
        '''
        # Initialize state values arbitrarily
        V = [0 for _ in range(self.m)]

    def dynamic_to_static(self, dyn: DynamicPolicy) -> StaticPolicy:
        '''
        Convert a dynamic to a static policy.
        '''
        pass


