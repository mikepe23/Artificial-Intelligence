# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def _computeBellmanSum(self, state, action, values_copy):
        transition_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        summation = 0
        for (successor_state, transition_prob) in transition_states_probs:
          reward = self.mdp.getReward(state, action, successor_state)
          successor_value = values_copy[successor_state]
          summation += transition_prob * (reward + (self.discount * successor_value))
        return summation

    def runValueIteration(self):
        states = self.mdp.getStates()
        

        # perform self.iterations number of iterations
        for _ in range(0, self.iterations):
          # compute value iteration for each MDP state
          values_copy = self.values.copy()
          for s in states:
            actions = self.mdp.getPossibleActions(s)
            self.values[s] = max([self._computeBellmanSum(s, action, values_copy) for action in self.mdp.getPossibleActions(s)], default=0)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        return self._computeBellmanSum(state, action, self.values)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        q_vals = util.Counter()
        for action in self.mdp.getPossibleActions(state):
          q_vals[action] = self.getQValue(state, action)
        return q_vals.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        # perform self.iterations number of iterations
        for i in range(0, self.iterations):
          # compute value iteration for each MDP state
          values_copy = self.values.copy()
          s = states[i % len(states)]
          actions = self.mdp.getPossibleActions(s)
          self.values[s] = max([self._computeBellmanSum(s, action, values_copy) for action in self.mdp.getPossibleActions(s)], default=0)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = util.Counter()
        allStates = self.mdp.getStates()
        # populate predecessors
        for s in allStates:        
          for a in self.mdp.getPossibleActions(s): 
            for ns in self.mdp.getTransitionStatesAndProbs(s, a):
              nextState = ns[0]
              if ns[1]:
                if not predecessors[nextState]:
                  predecessors[nextState] = {s}
                else:
                  predecessors[nextState].add(s)
        
        queue = util.PriorityQueue()
        for s in allStates:
          a = self.getAction(s)
          if a:
            Qval = self.getQValue(s,a)
            diff = math.fabs(self.values[s] - Qval)
            queue.update(s, -diff)
        for _ in range(0, self.iterations):
          if queue.isEmpty():
            return
          s = queue.pop()
          self.values[s] = max([self._computeBellmanSum(s, action, self.values) for action in self.mdp.getPossibleActions(s)], default=0)
          for p in predecessors[s]:
            a = self.getAction(p)
            if a:
              Qval = self.getQValue(p,a)
              diff = math.fabs(self.values[p] - Qval)    
              if diff > self.theta:
                queue.update(p, -diff)
        

