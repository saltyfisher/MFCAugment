import numpy as np

class Chromosome:
    def __init__(self, D, task_id, p_id):
        self.rnvec = np.random.rand(D)
        self.pbest = self.rnvec.copy()
        self.pbestFitness = np.inf
        self.task_id =  task_id # (genotype)--> decode to find design variables --> (phenotype)
        self.skill_factor = task_id
        self.p_id = p_id
        self.cost = np.inf

    def evaluate(self, Tasks, params):
        """单个染色体评估"""
        calls = 0
        params['task_id'] = self.task_id
        cost = Tasks[self.task_id].evaluate(self, params)
        self.cost = cost
        return self, calls, self.p_id

    @staticmethod
    def batch_evaluate(chromosomes, Tasks, params_list):
        """
        批量评估多个染色体
        :param chromosomes: Chromosome 对象列表
        :param Tasks: 任务列表
        :param params_list: 每个染色体对应的参数字典列表，或共享参数
        :return: 已评估的染色体列表、总函数调用次数、p_id 列表
        """
        if not chromosomes:
            return [], 0, []

        # 统一 task_id（假设同一批次属于同一任务）
        task_id = chromosomes[0].task_id
        for chrom in chromosomes:
            assert chrom.task_id == task_id, "所有染色体必须具有相同的 task_id 进行批量评估"

        # 构造输入向量 (N x D)
        rnvecs = np.array([chrom.rnvec for chrom in chromosomes])

        # 使用共享或独立的 params
        if isinstance(params_list, list):
            params = params_list[0]  # 简化为使用第一个参数集（可扩展）
        else:
            params = params_list

        params['task_id'] = task_id
        costs = Tasks[task_id].evaluate_batch(rnvecs, params)  # 假设任务支持 evaluate_batch

        # 分配代价
        calls = 0  # 可根据实际情况计数
        for i, chrom in enumerate(chromosomes):
            chrom.cost = costs[i]

        p_ids = [chrom.p_id for chrom in chromosomes]
        return chromosomes, calls, p_ids

    def pbestUpdate(self):
        if self.cost < self.pbestFitness:
            self.pbestFitness = self.cost
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