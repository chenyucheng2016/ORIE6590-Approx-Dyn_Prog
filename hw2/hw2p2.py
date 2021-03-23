import numpy as np
from copy import deepcopy
class Nmodel:
    def __init__(self):
        self.N = 10
        self.rho = 0.8
        self.gamma = 0.9
        self.h1 = 3
        self.h2 = 1
        self.lambda1 = 1.3*self.rho
        self.lambda2 = 0.4*self.rho
        m1 = 1.0
        m2 = 2.0
        m3 = 1.0
        self.eta1 = 1.0/m1
        self.eta2 = 1.0/m2
        self.eta3 = 1.0/m3
        self.actions = np.array([[0,0],[0,2],[0,3],[1,0],[1,2],[1,3]])
        self.B = self.eta1 + self.eta2 + self.eta3 + self.lambda1 + self.lambda2
        self.valueTable = np.zeros((self.N+1, self.N+1))
        self.actionDict = dict()
        self. iters = 1000
        self.tol = 0.1
    def reward(self, state):
        if state[0] < 0 or state[1] < 0 or state[0] > self.N or state[1] > self.N:
            print('some problem in state')
            return -1
        else:
            return self.h1*state[0] + self.h2*state[1]
    def transition_prob(self, state, action):
        ns_dist = dict()
        if state[0] < self.N:
            ns_dist[(state[0]+1, state[1])] = self.lambda1/self.B
        if state[1] < self.N:
            ns_dist[(state[0], state[1]+1)] = self.lambda2/self.B
        if action[0] == 0 and action[1] == 0:
            ns_dist[(state[0], state[1])] = 1 - self.sumUpRestEntries(ns_dist)
        elif action[0] == 0 and action[1] == 2:
            if state[0] > 0:
                ns_dist[(state[0]-1, state[1])] = self.eta2/self.B
            ns_dist[(state[0], state[1])] = 1 - self.sumUpRestEntries(ns_dist)
        elif action[0] == 0 and action[1] == 3:
            if state[1] > 0:
                ns_dist[(state[0], state[1] - 1)] = self.eta3 / self.B
            ns_dist[(state[0], state[1])] = 1 - self.sumUpRestEntries(ns_dist)
        elif action[0] == 1 and action[1] == 0:
            if state[0] > 0:
                ns_dist[(state[0]-1, state[1])] = self.eta1/self.B
            ns_dist[(state[0], state[1])] = 1 - self.sumUpRestEntries(ns_dist)
        elif action[0] == 1 and action[1] == 2:
            if state[0] > 0:
                ns_dist[(state[0]-1, state[1])] = (self.eta1 + self.eta2)/self.B
            ns_dist[(state[0], state[1])] = 1 - self.sumUpRestEntries(ns_dist)
        elif action[0] == 1 and action[1] == 3:
            if state[0] > 0:
                ns_dist[(state[0] - 1, state[1])] = self.eta1/self.B
            if state[1] > 0:
                ns_dist[(state[0], state[1] - 1)] = self.eta3 / self.B
            ns_dist[(state[0], state[1])] = 1.0 - self.sumUpRestEntries(ns_dist)
        return ns_dist
    def valueIter(self):
        for iter in range(self.iters):
            valueTable_cpy = deepcopy(self.valueTable)
            for i in range(self.N+1):
                for j in range(self.N+1):
                    state = [i,j]
                    maxVal = -1
                    optAct = []
                    for action in self.actions:
                        ns_dist = self.transition_prob(state, action)
                        val = self.reward(state)
                        for key in ns_dist.keys():
                            val = val + self.gamma*ns_dist[key]*valueTable_cpy[key[0], key[1]]
                        if val > maxVal:
                            maxVal =val
                            optAct = action
                    self.valueTable[i,j] = maxVal
                    self.actionDict[tuple(state)] = optAct
            if (np.max(abs(self.valueTable - valueTable_cpy))) < self.tol:
                break


    def sumUpRestEntries(self, ns_state):
        probSum = 0.0
        for key in ns_state.keys():
            probSum = probSum + ns_state[key]
        return probSum
    def init_policy(self):
        init_p = dict()
        for i in range(self.N+1):
            for j in range(self.N+1):
                state = tuple([i,j])
                r = np.random.randint(0,self.actions.shape[0])
                init_p[state] = self.actions[r]
        return init_p

    def policyIter(self):
        policy = self.init_policy()
        #policy eval
        for iter_out in range(self.iters):
            val_table = np.random.rand(self.N + 1, self.N + 1)
            for iter in range(self.iters):
                val_table_cpy = deepcopy((val_table))
                for i in range(self.N+1):
                    for j in range(self.N+1):
                        state = tuple([i,j])
                        action = policy[state]
                        ns_dist = self.transition_prob([i,j], action)
                        val = self.reward([i,j])
                        for key in ns_dist.keys():
                            val = val + self.gamma*ns_dist[key]*val_table_cpy[key[0], key[1]]
                        val_table[i,j] = val
                if np.max(abs(val_table - val_table_cpy)) < self.tol:
                    break

            #policy improvement
            p_changed = 0
            for i in range(self.N+1):
                for j in range(self.N+1):
                    state = tuple([i,j])
                    old_action = policy[state]
                    maxVal = -1
                    optAct = []
                    for action in self.actions:
                        ns_dist = self.transition_prob([i,j], action)
                        val = self.reward([i,j])
                        for key in ns_dist.keys():
                            val = val + self.gamma * ns_dist[key] * val_table[key[0], key[1]]
                        if val > maxVal:
                            maxVal = val
                            optAct = action
                    if optAct[0] != old_action[0] or optAct[1] != old_action[1]:
                        policy[state] = optAct
                        p_changed = 1
            if p_changed <= 0:
                break
            else:
                self.valueTable = val_table
                self.actionDict = policy
    def Explore_via_EpsilonGreedy(self, QTable, init_state):
        dataSize = 50000
        data = np.zeros((4, dataSize)) #2 for state, 1 for action, 1 for reward 2 + 1 + 1 = 4
        state = init_state
        epsilon = 0.4
        for i in range(dataSize):
            qs = QTable[state[0] ,state[1]]
            p_dist = np.zeros(self.actions.shape[0])
            for j in range(self.actions.shape[0]):
                p_dist[j] = epsilon/(self.actions.shape[0] - 1.0)
            p_dist[np.argmax(qs)] = 1.0 - epsilon
            action_ind = np.nonzero(np.random.multinomial(1, p_dist))[0][0]
            
            print(action_ind)



    def Explore_via_Boltzmann(self, QTable, init_state):
        dataSize = 50000
        data = np.zeros((4, dataSize)) #2 for state, 1 for action, 1 for reward 2 + 1 + 1 = 4

    def Q_learning(self):
        QTable = np.random.rand(self.N+1, self.N+1, self.actions.shape[0])
        self.Explore_via_EpsilonGreedy(QTable, [0,0])

            
if __name__ == "__main__":
    nmodel1 = Nmodel()
    nmodel1.Q_learning()
