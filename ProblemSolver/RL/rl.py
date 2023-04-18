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
import random
from typing import Tuple, Dict, List

# State - (j, l)
State = Tuple[int, int]
# map state to action
DynamicPolicy = Dict[State, int]
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

        self.v = q * r

        # States - see example.png for more clarity
        self.S = [(j, l) for j in range(1, self.m + 1) for l in range(1, n + 1) if j + l > self.n]
        
        # S_plus includes terminal states
        self.S_plus = [(j, l) for j in range(1, self.m + 2) for l in range(1, n + 1) if j + l > self.n]
        self.S_plus.append((self.m + 1, 0))

        # Actions - only 0, 1
        self.A = [0, 1]

        # Dynamics - see create_next_state()
        self.next_state_map = self.create_next_state()

    def create_next_state(self) -> Dict[Tuple[State, int], Tuple[List[float], List[Tuple[State, float]]]]:
        '''
        Maps the tuple (state, action) to a tuple of two lists. The first list is the probability
        corresponding to the outcome at the same index of the same list. The second list contains the
        reward and next state.

        Example:
        The return value ([0.2, 0.8], [((3, 2), 5), ((3, 3), 0)] means that there is a probability of 0.2
        to transition to (3, 2) and gain 5 reward and a probability of 0.8 to transition to (3, 3) and
        gain no reward.

        Generally, tabulate the dynamics of the system, representing the function p(s', r | s, a).
        Note that the dynamics_flowchart.png is 1-indexed, while
        Python is 0-indexed.
        '''
        next_state_map = {}

        for j, l in self.S:
            for a in self.A:
                if a == 1:
                    # the nurse chooses, the nurse doesn't choose
                    probs = [self.p[j - 1], 1 - self.p[j - 1]]
                    if l == 1:
                        state_rewards = [((self.m + 1, 0), self.v[j - 1]), ((j + 1, 1), 0)]
                    else:
                        state_rewards = [((j + 1, l - 1), self.v[j - 1]), ((j + 1, l), 0)]
                else:
                    probs = [1]
                    state_rewards = [((j + 1, l), 0)]

                next_state_map[((j, l), a)] = (probs, state_rewards)

        return next_state_map

    def step(self, s: int, a: int) -> Tuple[State, float, bool]:
        '''
        Randomly advance from a current state with a given action.
        Used for MC sampling.

        Returns a tuple of:
        - s' - the state tuple [j, l]
        - r - the reward
        - terminated - whether or not we have reached the terminal state
        '''
        probs, state_rewards = self.next_state_map[(s, a)]
        idx = random.choices(range(len(probs)), weights=probs)[0]

        next_state, reward = state_rewards[idx]
        next_j, next_l = next_state
        terminated = next_j == self.m + 1 or next_l == 0

        return (next_state, reward, terminated)
    
    def iteration(self, gamma=1, max_iters=100, abs_tol=0.001) -> Tuple[DynamicPolicy, Dict[State, float]]:
        '''
        Perform value iteration on the MDP
        From Sutton, Barto 2017, Chapter 4.4
        Return a dictionary mapping state to deterministic action and the learned state-value function.

        - gamma is a measure of farsightedness between 0 and 1.
        - max_iters determines the number of iterations.
        - abs_tol determines the absolute tolerance of the state value
          estimates.
        '''
        def state_action(s: State, a: int, V: List[float]):
            '''
            The state-action function. The expected value of
            discounted reward.
            '''
            expected_reward = 0
            probs, state_rewards = self.next_state_map[(s, a)]

            for prob, state_reward in zip(probs, state_rewards):
                new_state, reward = state_reward
                expected_reward += prob * (reward + gamma * V[new_state])

            return expected_reward

        # Initialize state values arbitrarily
        V = {s: 0 for s in self.S_plus}
        # Value estimation
        for _ in range(max_iters):
            delta = 0

            for s in self.S:
                v = V[s]
                # The policy improvement theorem
                # V(s) = max_a q(s, a)
                V[s] = max(state_action(s, a, V) for a in self.A)
                delta = max(delta, abs(v - V[s]))

            # Consider using relative tolerance
            # or the Bellman error, the difference in the LHS and RHS of the
            # Bellman equation
            if delta < abs_tol: break

        # Get the optimal policy
        optimal_policy = {}

        for s in self.S:
            optimal_policy[s] = max(self.A, key=lambda a: state_action(s, a, V))

        return optimal_policy, V
    
    def create_probability(self) -> Dict[State, float]:
        '''
        For each state in S+, return the probability of reaching that state.
        '''


    def dynamic_to_static(self, dyn: DynamicPolicy) -> StaticPolicy:
        '''
        Convert a dynamic to a static policy.
        '''
        pass


