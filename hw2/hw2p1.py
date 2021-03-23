import numpy as np
import copy
class GridWorld:
    def __init__(self):
        self.r0 = -1 + 100
        self.dim = 5
        self.grid = np.random.rand(self.dim, self.dim)
        self.actions = np.zeros((self.dim, self.dim))
        self.grid[0][1] = 10
        self.grid[0][3] = 5
    def valueIteration(self):
        iter = 5000
        for k in range(iter):
            grid_cpy = copy.deepcopy(self.grid)
            for i in range(self.dim):
                for j in range(self.dim):
                    if (i == 0 and j == 1) or (i == 0 and j == 3):
                        continue
                    else:
                        s_prime_list = []
                        s_prime_list.append(grid_cpy[np.maximum(i - 1, 0), j]) #north
                        s_prime_list.append(grid_cpy[np.minimum(i + 1, self.dim - 1), j])#soth
                        s_prime_list.append(grid_cpy[i, np.maximum(j - 1, 0)])#west
                        s_prime_list.append(grid_cpy[i, np.minimum(j + 1, self.dim - 1)])#est
                        self.grid[i][j] = self.r0 + np.max(s_prime_list)
                        cmp_list = []
                        if i - 1 >= 0:
                            cmp_list.append(self.grid[i - 1, j])
                        else:
                            cmp_list.append(-1)
                        if i + 1 < self.dim:
                            cmp_list.append(self.grid[i + 1, j])
                        else:
                            cmp_list.append(-1)
                        if j - 1 >= 0:
                            cmp_list.append(self.grid[i, j - 1])
                        else:
                            cmp_list.append(-1)
                        if j + 1 < self.dim:
                            cmp_list.append(self.grid[i, j + 1])
                        else:
                            cmp_list.append(-1)

                        self.actions[i][j] = np.argmax(cmp_list)
    def printACtionTable(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if (i == 0 and j == 1) or (i == 0 and j == 3):
                    print('o ', end="")
                    continue
                if self.actions[i,j] == 0:
                    print('n ', end="")
                elif self.actions[i,j] == 1:
                    print('s ', end="")
                elif self.actions[i, j] == 2:
                    print('w ', end="")
                elif self.actions[i, j] == 3:
                    print('e ', end="")
            print("\n")

if __name__ == "__main__":
    grid = GridWorld()
    grid.valueIteration()
    print(grid.grid)
    grid.printACtionTable()
