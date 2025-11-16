import numpy as np

class Chromosome:
    def __init__(self, D, no_of_tasks, p_id):
        self.rnvec = np.random.rand(D)  # (genotype)--> decode to find design variables --> (phenotype)
        self.pbest = self.rnvec.copy()
        self.pbestFitness = np.inf
        self.velocity = 0.1 * self.pbest
        self.factorial_costs = np.inf * np.ones(no_of_tasks)
        self.factorial_ranks = np.zeros(no_of_tasks)
        self.scalar_fitness = 0
        self.skill_factor = -1
        self.no_of_tasks = no_of_tasks
        self.p_id = p_id

    def evaluate(self, Tasks, no_of_tasks, params):
        calls = 0
        funcCount = 1
        if self.skill_factor == -1:
            for i in range(no_of_tasks):
                params.update({'task_id':i})
                cost = Tasks[i].evaluate(self, params)
                self.factorial_costs[i] = cost
                calls += funcCount
        else:
            self.factorial_costs[:] = np.inf
            for i in range(no_of_tasks):
                if self.skill_factor == i:
                    params.update({'task_id':i})
                    cost = Tasks[self.skill_factor].evaluate(self, params)
                    self.factorial_costs[self.skill_factor] = cost                    
                    calls = funcCount
                    break
        return self, calls, self.p_id

    def evaluate_SOO(self, Task, p_il, options):
        cost, rnvec, funcCount = fnceval(Task, self.rnvec, p_il, options)
        self.factorial_costs = cost
        self.rnvec = rnvec
        calls = funcCount
        return self, calls

    def pbestUpdate(self):
        if self.factorial_costs[self.skill_factor] < self.pbestFitness:
            self.pbestFitness = self.factorial_costs[self.skill_factor]
            self.pbest = self.rnvec.copy()
        return self
    def crossover(self, p1, p2, cf):
        self.rnvec = 0.5 * ((1 + cf) * p1.rnvec + (1 - cf) * p2.rnvec)
        self.rnvec = np.clip(self.rnvec, 0, 1)
        return self

    def mutate(self, p, D, sigma):
        rvec = np.random.normal(0, sigma, D)
        self.rnvec = p.rnvec + rvec
        self.rnvec = np.clip(self.rnvec, 0, 1)
        return self
# 注意：你需要自己实现 fnceval 函数，并确保 Tasks 的结构与 MATLAB 版本一致。