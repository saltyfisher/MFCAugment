import numpy as np
from scipy.optimize import minimize

class Particle:
    def __init__(self, D, no_of_tasks):
        self.rnvec = np.random.rand(D)  # (genotype)--> decode to find design variables --> (phenotype)
        self.pbest = self.rnvec.copy()
        self.pbestFitness = np.inf
        self.velocity = 0.1 * self.pbest
        self.factorial_costs = np.inf * np.ones(no_of_tasks)
        self.factorial_ranks = np.zeros(no_of_tasks)
        self.scalar_fitness = 0
        self.skill_factor = -1
        self.no_of_tasks = no_of_tasks

    def evaluate(self, Tasks, no_of_tasks, params):
        calls = 0
        funcCount = 1
        if self.skill_factor == -1:
            for i in range(no_of_tasks):
                params.update({'task_id':i})
                cost = Tasks[i].evaluate(self.rnvec, params)
                self.factorial_costs[i] = cost
                calls += funcCount
        else:
            self.factorial_costs[:] = np.inf
            for i in range(no_of_tasks):
                if self.skill_factor == i:
                    params.update({'task_id':i})
                    cost = Tasks[self.skill_factor].evaluate(self.rnvec, params)
                    self.factorial_costs[self.skill_factor] = cost
                    calls = funcCount
                    break
        return calls

    def evaluate_SOO(self, Task, p_il, options):
        cost, self.rnvec, funcCount = fnceval(Task, self.rnvec, p_il, options)
        calls = funcCount
        return self, calls

    # position update
    def positionUpdate(self):
        self.rnvec = self.rnvec + self.velocity
        self.rnvec = np.clip(self.rnvec, 0, 1)
        return self

    # pbest update
    def pbestUpdate(self):
        if self.factorial_costs[self.skill_factor] < self.pbestFitness:
            self.pbestFitness = self.factorial_costs[self.skill_factor]
            self.pbest = self.rnvec.copy()
        return self

    def pbestUpdate_SOO(self):
        if self.factorial_costs < self.pbestFitness:
            self.pbestFitness = self.factorial_costs
            self.pbest = self.rnvec.copy()
        return self

    # velocity update
    def velocityUpdate(self, gbest, rmp, w1, c1, c2, c3):
        len_velocity = len(self.velocity)
        idx = list(range(self.no_of_tasks))
        idx.remove(self.skill_factor)
        rand_idx = int(np.random.choice(idx, 1)[0])
        if np.random.rand() < rmp:
            self.velocity = w1 * self.velocity + \
                            c1 * np.random.rand(len_velocity) * (self.pbest - self.rnvec) + \
                            c2 * np.random.rand(len_velocity) * (gbest[self.skill_factor] - self.rnvec) + \
                            c3 * np.random.rand(len_velocity) * (gbest[rand_idx] - self.rnvec)
            if np.random.rand() < 0.5:
                self.skill_factor = rand_idx
        else:
            self.velocity = w1 * self.velocity + \
                            c1 * np.random.rand(len_velocity) * (self.pbest - self.rnvec) + \
                            c2 * np.random.rand(len_velocity) * (gbest[self.skill_factor] - self.rnvec)
        return self

    def velocityUpdate_SOO(self, gbest, w1, c1, c2):
        len_velocity = len(self.velocity)
        self.velocity = w1 * self.velocity + \
                        c1 * np.random.rand(len_velocity) * (self.pbest - self.rnvec) + \
                        c2 * np.random.rand(len_velocity) * (gbest - self.rnvec)
        return self

# Example usage of the Particle class
# no_of_tasks should be defined before creating a Particle instance
# fnceval should be defined to evaluate the cost function