# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from collections import Counter
from learningAgents import ValueEstimationAgent

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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    for i in range(iterations):
        iter_vals = self.values.copy()
        for s in self.mdp.getStates():
            a_sums = Counter()
            for a in self.mdp.getPossibleActions(s):
                for s_prime,p in self.mdp.getTransitionStatesAndProbs(s,a):
                    a_sums[a] += self.bellmanUpdate(p,s,a,s_prime,iter_vals)
            self.values[s] = 0 if len(a_sums.values()) == 0 else max(a_sums.values())

  def bellmanUpdate(self,p,s,a,s_prime,iter_vals):
      immediate_r = self.mdp.getReward(s,a,s_prime)
      future_r = self.discount * iter_vals[s_prime]
      return p * (immediate_r + future_r)

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    q = self.values[state]
    for s_prime,p in self.mdp.getTransitionStatesAndProbs(state,action):
        q += p * self.discount * self.values[s_prime]
    return q

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    a_sums = Counter()
    for a in self.mdp.getPossibleActions(state):
        for s_prime,p in self.mdp.getTransitionStatesAndProbs(state,a):
            a_sums[a] += self.bellmanUpdate(p,state,a,s_prime,self.values)
    return 0 if len(a_sums.values()) == 0 else max(a_sums.values())

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
