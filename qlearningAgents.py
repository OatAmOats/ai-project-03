# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q = util.Counter()
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.Q: #if the state action tuplet doesn't have a Q value yet, set it to zero.
            self.Q[(state, action)] = 0.0
            return 0.0
        return self.Q[(state, action)] #otherwise just return the Q-Value
        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        maxValue = -float('inf')
        actions = self.getLegalActions(state)
        if actions == () or len(actions) == 0: #if there are no possible actions return 0.0
            return 0.0
        for a in actions: #otherwise iterate through possible actions for the best Q-Value
            maxValue = max(maxValue, self.getQValue(state, a))
        return maxValue
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None. 
        """
        "*** YOUR CODE HERE ***"
        
        action = None #initialize action as None
        actions = self.getLegalActions(state)
        bestVal = -float('inf') #and the best value as -infinity
        if len(actions) == 0: #if there are no possible actions, return None
            return None
        for a in actions: #for every action, get the Q-Value, and keep store the current best action in action, until done
            val = self.getQValue(state, a)
            if val > bestVal:
                bestVal = val
                action = a
        
        # if bestVal == -float('inf'): 
        #     return random.choice(actions)
        # if bestVal == 0.0:
        #     return action
        return action
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        p = self.epsilon
        if len(legalActions) == 0: #if there are no possible actions ,return None
            return action
        if util.flipCoin(p): #with a probability of epsilon, choose a random action
            return random.choice(legalActions)
        else: #the rest of the time, pick action based on Q-values
            return self.computeActionFromQValues(state)
        util.raiseNotDefined()
        

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #if there was a state before the current one:
        if state != None:
            #update Q value for previous state and the action taken using equation
            self.Q[(state, action)] = self.getQValue(state, action) +  self.alpha * (reward + (self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action))
        else:
            return 0.0 
        return (self.getQValue(state, action))
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        weights = self.getWeights().copy() #get the weights
        features = self.featExtractor.getFeatures(state, action) #and the features
        
        return weights * features
        
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action) #get the features
        weights = self.getWeights().copy() #create a copy of the (old) weights to work with
        
        for feature in features: #for every feature
            difference = reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, action)
            weights[feature] = weights[feature] + (self.alpha * difference * features[feature]) #update the weight based on the old weight, the feature, and the difference
        self.weights = weights #update self.weights
        return 
        util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
