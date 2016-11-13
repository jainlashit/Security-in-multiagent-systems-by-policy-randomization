import csv
import cvxopt
from cvxopt import solvers
import numpy as np
       
A_mat = []
R_mat = []
alpha = []
G = []
h = []
high_entropy_policy = []

class State:

    def __init__(self, ind, name, actions):
        self.index = ind
        self.name = name
        self.possibleActions = actions
        self.transition = []
        self.reward = []
        self.terminating = False
        self.utility = 0

    def __repr__(self):
        return "Index: " + str(self.index) + " Name: " + self.name + " Actions: " + str(self.possibleActions)

    #def __str__(self):
     #   print "Index: " + str(self.index) + " Name: " + self.name + " Actions: " + str(self.possibleActions)

    def modifyActions(self, actions):
        self.possibleActions = actions

    def setTransition(self, tran):
        self.transition = tran

    def getTransition(self):
        return self.transition

    def setReward(self, reward):
        self.reward = reward

    def getReward(self):
        return self.reward

    def getIndex(self):
        return self.index

    def getPossibleActions(self):
        return self.possibleActions

    def setPossibleActions(self, act):
        self.possibleActions = act

    def isTerminating(self):
        return self.terminating

    def setTerminating(self, term):
        self.terminating = term
        if term == True:
            self.possibleActions = []

    def setUtility(self, util):
        self.utility = util

    def getUtility(self):
        return self.utility

class Action:

    def __init__(self, ind, name):
        self.index = ind
        self.name = name

    def __repr__(self):
        return "Index: " + str(self.index) + " Name: " + self.name

    def getIndex(self):
        return self.index


class MDP:

    def __init__(self, numberOfStates, numberOfActions):
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.numberOfOptions = 0
        self.states = []
        self.actions = []
        self.options = []

    # Define Action
    def initializeActions(self):
        for i in xrange(0, self.numberOfActions):
            a = Action(i, str("a" + str(i)))
            self.actions.append(a)

    # Define States
    def initializeStates(self):
        for i in xrange(0, self.numberOfStates):
            x = State(i, str("s" + str(i)), self.actions[0:self.numberOfActions-1])
            self.states.append(x)
        self.states[3].setTerminating(True)
        self.states[3].setUtility(1)
        self.states[3].setPossibleActions([self.actions[self.numberOfActions-1]])
        self.states[7].setTerminating(True)
        self.states[7].setUtility(-1)
        self.states[7].setPossibleActions([self.actions[self.numberOfActions - 1]])

    # Leave one line space after each transition table for each action in the data file.
    # TransitionFunction For Acti
    def autoTransitionFunction(self, gamma=1):
        for s in self.states:
            s.setTransition([])
        stateIndex = 0
        actionIndex = 0
        with open('transitionData', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) == 0:
                    stateIndex = 0
                    actionIndex = actionIndex + 1
                    continue
                for sp in xrange(0, self.numberOfStates):
                    triple = (actionIndex, sp, float(row[sp])*gamma)
                    self.states[stateIndex].getTransition().append(triple)
                stateIndex += 1

    # RewardFunctions For Actions
    def autoRewardFunction(self):

        tosend = []
        stateIndex = 0
        with open('rewardData', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row)==0:
                    continue
                for ap in xrange(0, self.numberOfActions):
                    triple = (ap, float(row[ap]))
                    tosend.append(triple)
                self.states[stateIndex].setReward(tosend)
                tosend = []
                stateIndex += 1


    def generateLPAc(self):
        decisionvar = []
        for x in self.states:
            triple = []
            for y in self.states:
                triplet = []
                for a in y.possibleActions:
                    if x.getIndex() == y.getIndex():
                        triplet.append(float(1))
                    else:
                        triplet.append(float(0))
                triple.append(triplet)
            decisionvar.append(triple)

        for x in self.states:
            incoming = []
            for s in self.states:
                for t in s.transition:
                    if t[1]==x.getIndex() and t[2]!=0:
                        incoming.append((s, t[0], t[2]))

            for h in incoming:
                decisionvar[x.getIndex()][h[0].getIndex()][h[1]] -= float(h[2])

        for x in decisionvar:
            lit = []
            for t in x:
                lit.extend(t)
            A_mat.append(lit)

        # for x in self.states:
        #     for r in x.reward:
        #         R_mat.append(float(r[1]))
        for x in self.states:
            for y in x.possibleActions:
                for r in x.reward:
                    if r[0]==y.getIndex():
                        R_mat.append(r[1])
        
        #print R_mat
        # for x in A_mat:
        #     print x


class Driver:
    a = MDP(12, 5)
    a.initializeActions()
    a.initializeStates()
    a.autoTransitionFunction()
    a.autoRewardFunction()
    a.generateLPAc()
    # print a.states[0].transition

def solveLP(beta):
    global R_mat, A_mat, alpha, G, h, high_entropy_policy
    G = np.add(beta * high_entropy_policy,  -1 * np.identity(np.shape(R_mat)[0]))
    A = cvxopt.matrix(A_mat)
    b = cvxopt.matrix(alpha)
    c = cvxopt.matrix(R_mat)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    sol = solvers.lp(c, G, h, A, b)
    x_mat = np.array(sol['x'])
    optimal_solution  = np.dot(R_mat.T, x_mat)
    return x_mat, -1*optimal_solution[0][0]

def brlp(Emin):
    beta_low = 0.0
    beta_high = 1.0
    beta = (beta_low + beta_high)/2
    epsilon = 0.01
    x_mat, optimal_solution = solveLP(beta)
    while abs(optimal_solution - Emin) > epsilon :
        if optimal_solution > Emin:
            beta_low = beta
        else:
            beta_high = beta
        beta = (beta_low + beta_high)/2
        x_mat, optimal_solution = solveLP(beta)
        print("========================================================")
        print("Optimal Solution :", optimal_solution)
        print("========================================================")

    return x_mat

def pre_lpSolver():
    global R_mat, A_mat, alpha, G, h, high_entropy_policy
    count = 0
    high_entropy_policy = np.zeros((len(R_mat), len(R_mat)))
    for i in range(len(A_mat)):
        if(i == 3 or i == 7):
            high_entropy_policy[count][count] = 1.0
            count += 1
        else:
            for j in range(4):
                for k in range(4):
                    high_entropy_policy[count + j][count + k] = 0.25
            count += 4

    R_mat = -1 * np.array(R_mat)[np.newaxis].T
    A_mat = np.array(A_mat)
    alpha = np.zeros((np.shape(A_mat)[0], 1))
    h = np.zeros((np.shape(R_mat)[0], 1))
    alpha[8][0] = 1.0

pre_lpSolver()
brlp(0.4)
# R_mat 1x42
# A_mat 12x42