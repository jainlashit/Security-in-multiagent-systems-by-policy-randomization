import csv
import cvxopt
from cvxopt import solvers
import numpy as np
       
A_mat = []
R_mat = []
Ran_X = []  

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

        for x in self.states:
            total=float(len(x.possibleActions))
            for i in xrange(0,len(x.possibleActions)):
                Ran_X.append(float(1/total));
           
        #print Ran_X
        
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



def calc_entropy(x_mat):
    count = 0
    policy = np.zeros(np.shape(R_mat)[0])
    for i in range(np.shape(A_mat)[0]):
        if i == 3 or i == 7:
            policy[count] = 1
            count += 1
        else:
            temp = 0
            for j in range(4):
                temp += x_mat[count + j]
            for j in range(4):
                policy[count + j] = x_mat[count + j]/temp
            count += 4
    entropy = 0
    # print(policy)
    for i in range(len(policy)):
        entropy += -1 * policy[i] * np.log(policy[i])

    print('Using additive entropy matrix, entropy obtained', entropy)


# R_mat 1x42
# A_mat 12x42
R_mat = -1 * np.array(R_mat)[np.newaxis].T
A_mat = np.array(A_mat)
alpha = np.zeros((np.shape(A_mat)[0], 1))
G = -1 * np.identity(np.shape(R_mat)[0])
h = np.zeros((np.shape(R_mat)[0], 1))
alpha[8][0] = 1.0 

A = cvxopt.matrix(A_mat)
b = cvxopt.matrix(alpha)
c = cvxopt.matrix(R_mat)
G = cvxopt.matrix(G)
h = cvxopt.matrix(h)
sol = solvers.lp(c, G, h, A, b)
#print(sol['x'])
x_mat = np.array(sol['x'])
Ran_X = np.array(Ran_X)[np.newaxis].T
optimal_solution  = -1 * np.dot(R_mat.T, x_mat)[0][0]
average_solution = -1 * np.dot(R_mat.T, Ran_X)[0][0]
Emin  = 0.7
beta = (optimal_solution - Emin)/(optimal_solution - average_solution)
X_crlp = np.add((1-beta)*x_mat, beta*Ran_X)

crlp_solution = -1 * np.dot(R_mat.T, X_crlp)[0][0]
calc_entropy(X_crlp)

print "Reward obtained :", crlp_solution