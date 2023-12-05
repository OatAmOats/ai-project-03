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


import mdp, util

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
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
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

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp #for easier access
        maxError = 1 #don't think this matters
        delta = 0 #initialize delta as zero
        newDelta = float('inf') #and new delta as infinity (so that delta = 0 doens't mess w it later)
        states = mdp.getStates() #get the states
        n = 1 #initalize iteration as zero
        while n <= self.iterations:
            delta = 0
            newUtil = self.values.copy() #store a copy of the current utilities
            for state in states: #for each state
                bestVal = -float('inf') #initialize the best value as negative infinity
                for action in mdp.getPossibleActions(state): #update bestVal to be the highest value
                    statesAndProbs = mdp.getTransitionStatesAndProbs(state, action)
                    actionVal = 0.0
                    for x in statesAndProbs:
                        actionVal += x[1] * self.values[x[0]]
                    bestVal = max(bestVal, actionVal)
                if mdp.getPossibleActions(state) == (): #if there are no possible actions, bestVal = 0.0
                    bestVal = 0.0
                newUtil[state] = mdp.getReward(state, None, None) + self.discount * bestVal #update the utility of the state in newUtil

                if abs(newUtil[state] - self.values[state]) > delta: #update delta
                    delta = abs(newUtil[state] - self.values[state]) 
                newDelta = delta
            self.values = newUtil #update self.values to be equal to newUtil
            if newDelta < 0.01 * (1-self.discount)/self.discount: #end the iteration if newDelta < etc
                n = 100
            n += 1 #increase the iteration counter
        return self.values 


        
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
        "*** YOUR CODE HERE ***"
        qValue = 0.0
        for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):  #add the value * probabiltiy for each potential state to q
            qValue += self.values[newState] * prob
        return self.mdp.getReward(state, None, None) + self.discount * qValue #return reward + discount * q
        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None #initialize best action and best value
        bestValue = -float('inf')
        for action in self.mdp.getPossibleActions(state):
            actionValue = self.computeQValueFromValues(state, action)
            bestValue = max(bestValue, actionValue) #whenever bestValue is updated, update bestAction too
            if actionValue == bestValue:
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
