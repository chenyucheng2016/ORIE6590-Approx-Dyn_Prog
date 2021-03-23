import numpy as np

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
    def transition_prob(self, state, action):
        if action == [0,0]:
        elif action == [0,2]:
            
if __name__ == "__main__":
    nmodel = Nmodel()