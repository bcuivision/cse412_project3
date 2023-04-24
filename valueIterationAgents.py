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

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        values = self.values.copy()

        for i in range(0,self.iterations): 
            
            for s in self.mdp.getStates():
                max = float("-inf")

                for a in self.mdp.getPossibleActions(s):

                    qValue = self.computeQValueFromValues(s,a)

                    if (qValue >= max):
                        max = qValue
                        values[s] = qValue

            #update values
            self.values = values.copy()    

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

        qValue = 0

        for (s,p) in self.mdp.getTransitionStatesAndProbs(state,action):
            r = self.mdp.getReward(state,action,s)
            v = self.discount * self.getValue(s)
            qValue += p * (r + v)
            
        return qValue
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        #check if state = terminal
        if self.mdp.isTerminal(state):
            return 
        
        else:
            actionDict = util.Counter()

            for a in self.mdp.getPossibleActions(state):
                actionDict[a] = self.computeQValueFromValues(state,a)

            return actionDict.argMax()    

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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in range(0,self.iterations): 
            s = states[i % len(states)]

            if not self.mdp.isTerminal(s):
                self.values[s] = self.computeQValueFromValues(s,self.getAction(s))


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
        "*** YOUR CODE HERE ***"

        #Compute predecessors of all states.
        predecessors = {}
        states = self.mdp.getStates()

        for s in states:
            if not self.mdp.isTerminal(s):
                for a in self.mdp.getPossibleActions(s):
                    for (successor, p) in self.mdp.getTransitionStatesAndProbs(s, a):
                        
                        if successor in predecessors:
                            predecessors[successor].add(s)
                        else:
                            predecessors[successor] = {s}

        #Initialize an empty priority queue.
        queue = util.PriorityQueue()

        for s in states:
            if not self.mdp.isTerminal(s):
                qValues = []

                for a in self.mdp.getPossibleActions(s):
                    qValues.append(self.computeQValueFromValues(s,a))
                
                #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s 
                diff = abs(self.values[s] - max(qValues))

                #Push s into the priority queue with priority -diff (note that this is negative).
                queue.update(s, -diff)
        
        for i in range(0, self.iterations):
            #if queue = empty, terminate.
            if queue.isEmpty():
                return

            #Pop a state s off the priority queue.
            s = queue.pop()

            #Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                qValues = []

                for a in self.mdp.getPossibleActions(s):
                    qValues.append(self.computeQValueFromValues(s,a))
                
                self.values[s] = max(qValues)

            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    qValues = []

                    for a in self.mdp.getPossibleActions(p):
                        qValues.append(self.computeQValueFromValues(p,a))
                    
                    #Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p
                    diff = abs(self.values[p] - max(qValues))

                    #If diff > theta, push p into the priority queue with priority -diff
                    if diff > self.theta:
                        queue.update(p, -diff)
