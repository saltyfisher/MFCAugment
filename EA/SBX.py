import numpy as np
import time
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from EA.ChromosomeSingle import Chromosome
from tqdm import tqdm
import joblib
import pickle
import random

def evaluate_population_batch(population, Tasks, params, fnceval_calls, rep):
    """
    批量评估种群中所有个体的函数
    
    参数:
    population: 要评估的种群（个体列表）
    Tasks: 任务列表
    params: 参数字典
    fnceval_calls: 函数评估调用计数器
    rep: 当前重复次数
    
    返回:
    更新后的种群和评估调用计数
    """
    pop = len(population)
    task_ids = np.array([ind.task_id for ind in population])
    no_of_tasks = len(Tasks)
    # 按任务分组进行批量评估
    for task_id in range(no_of_tasks):
        # 获取属于当前任务的所有个体索引
        task_indices = [i for i in range(pop) if population[i].task_id == task_id]
        
        if task_indices:  # 如果存在属于该任务的个体
            # 收集这些个体的策略
            policies = [population[i] for i in task_indices]
            
            # 设置参数中的任务ID
            params['task_id'] = task_id
            
            # 批量评估
            from core.MFCAugment import evalFuncBatch
            losses = evalFuncBatch(policies, params)
            
            # 将评估结果分配给对应的个体
            for idx, loss in zip(task_indices, losses):
                population[idx].cost = loss
                fnceval_calls[rep] += 1
                
    return population, fnceval_calls

def SBX(Tasks, options, params, writer):
    pop = options['popsize']
    gen = options['maxgen']
    rmp = options['rmp']
    reps = options['reps']
    if pop % 2 != 0:
        pop += 1
    
    no_of_tasks = len(Tasks)
    D = np.zeros(no_of_tasks)
    for i in range(no_of_tasks):
        D[i] = Tasks[i].dims
    D_multitask = int(np.max(D))
    
    options = {'disp': False, 'maxiter': 2}  # 个体学习的设置
    recorder = [{'Pops':[],'Fval':[],'SkillFactor':[]} for _ in range(reps)]
    fnceval_calls = np.zeros(reps)
    pop = pop*no_of_tasks
    EvBestFitness = np.zeros((no_of_tasks * reps, gen))  # 到目前为止找到的最佳适应度
    TotalEvaluations = np.zeros((reps, gen))  # 到目前为止的任务评估总数
    bestobj = np.inf * np.ones(no_of_tasks) # 到目前为止找到的任务最优目标值
    bestPop = np.zeros((reps, pop), dtype=object)
    bestInd_data = np.zeros((reps, no_of_tasks), dtype=object)
    gbest = np.zeros((no_of_tasks, D_multitask))
    # with joblib.Parallel(n_jobs=1, backend='threading') as parallel:
    for rep in range(reps):
        population = [Chromosome(D_multitask, i%no_of_tasks, i) for i in range(pop)]
        task_ids = np.array([i%no_of_tasks for i in range(pop)])
        st = time.time()
        # 新的批量评估方式（可选）
        population, fnceval_calls = evaluate_population_batch(population, Tasks, params, fnceval_calls, rep)
        
        # 原始的逐个评估方式
        # results = parallel(joblib.delayed(population[i].evaluate)(Tasks, params) for i in range(pop))
        # for i in range(pop):
        #     calls_per_individual[i] = population[i].evaluate(Tasks, no_of_tasks, params)
        # for r in results:
        #     fnceval_calls[rep] += r[1]
        #     population[r[2]] = r[0]

        TotalEvaluations[rep, 0] = fnceval_calls[rep]
        print(f'Initializing:{time.time()-st:.2f}')
        recorder[rep]['Pops'].append([population[i].rnvec for i in range(pop)])
        recorder[rep]['Fval'].append([population[i].cost for i in range(pop)])
        for i in range(no_of_tasks):      
            costs = []
            pops = []
            idx = np.where(task_ids==i)[0]
            for j in idx:
                costs.append(population[j].cost)
                pops.append(population[j])
            sorted_indices = np.argsort(costs)
            pops = [pops[i] for i in sorted_indices]  # 根据当前任务的factorial_cost重新排序种群
            bestobj[i] = pops[0].cost
            bestInd_data[rep, i] = pops[0]
            gbest[i] = pops[0].rnvec
            EvBestFitness[i + 2 * (rep), 0] = bestobj[i]
        
        ite = 1
        mu = 10
        sigma = 0.02
        noImpove = np.zeros(no_of_tasks)
        converge = np.zeros(no_of_tasks)
        while ite <= gen:
            st = time.time()               
            if all(converge >= 30):
                break
            for i in range(pop):
                population[i].pbestUpdate()
            count = 0
            child = np.zeros(pop, dtype=object)
            for i in range(no_of_tasks):
                idx = np.where(task_ids==i)[0]
                inorder = np.random.permutation(np.arange(len(idx)))
                inorder = idx[inorder]
                for j in range(len(idx)//2):
                    p1_idx = inorder[j]
                    p2_idx = inorder[j+len(idx)//2]
                    p1 = population[p1_idx]
                    p2 = population[p2_idx]
                    child[p1_idx] = Chromosome(D_multitask, p1.task_id, p1_idx)
                    child[p2_idx] = Chromosome(D_multitask, p2.task_id, p2_idx)
                    # if (np.random.rand()<0.5):
                    if False:
                        u = np.random.rand(D_multitask)
                        cf = np.zeros(D_multitask)
                        cf[u<=0.5]=(2*u[u<=0.5])**(1/(mu+1))
                        cf[u>0.5]=(2*(1-u[u>0.5]))**(-1/(mu+1))
                        child[p1_idx] = child[p1_idx].crossover(p1,p2,cf)
                        child[p2_idx] = child[p2_idx].crossover(p2,p1,cf)
                    else:
                        child[p1_idx] = child[p1_idx].mutate(p1,D_multitask,sigma)
                        child[p2_idx] = child[p2_idx].mutate(p2,D_multitask,sigma)
                    count = count+2

            # 新的批量评估方式（可选）
            child, fnceval_calls = evaluate_population_batch(child, Tasks, params, fnceval_calls, rep)
            
            # 原始的逐个评估方式
            # results = parallel(joblib.delayed(child[i].evaluate)(Tasks, params) for i in range(pop))

            # for r in results:
            #     fnceval_calls[rep] += r[1]
            #     child[r[2]] = r[0]                 
            
            for i in range(no_of_tasks):
                idx = np.where(task_ids==i)[0]
                pops = [population[j] for j in idx]  
                chs = [child[j] for j in idx]  
                pop_size = len(idx)
                intpopulation = np.zeros(2*pop_size, dtype=object)
                intpopulation[:pop_size] = pops
                intpopulation[pop_size:] = chs
                costs = [p.cost for p in intpopulation]
                sorted_indices = np.argsort(costs)
                intpopulation = [intpopulation[i] for i in sorted_indices]
                if intpopulation[0].cost <= bestobj[i]:
                    bestobj[i] = intpopulation[0].cost                   
                    bestInd_data[rep, i] = intpopulation[0]
                    noImpove[i] = 0
                else:
                    noImpove[i] += 1
                EvBestFitness[i + 2 * (rep), ite-1] = bestobj[i]

                # ranks = np.argsort([p.cost for p in intpopulation])
                # intpopulation = [intpopulation[r] for r in ranks]
                # pops = intpopulation[:pop_size]
                count = 0
                while count < pop_size:
                    costs = [p.cost for p in intpopulation]
                    scaler = MinMaxScaler()
                    costs = scaler.fit_transform(np.array(costs).reshape(-1, 1))
                    cumsum_p = np.cumsum(costs)
                    cumsum_p = cumsum_p-np.random.rand()
                    if any(cumsum_p>0) is False:
                        x = np.random.randint(pop_size)
                    else:
                        x = list(cumsum_p>0).index(True)
                    pops[count] = intpopulation[x]
                    count = count + 1

                for k, j in enumerate(idx):
                    population[j] = pops[k]

            bestobj_str = [f'{o:.4f}' for o in bestobj]
            print(f'Time:{time.time()-st:.2f} Generation: {ite} Repeat: {rep} Best Fitness: {bestobj_str}')
            for k in range(no_of_tasks):
                writer[k].add_scalar(f'Best_Fitness/Rep{rep}', bestobj[k], ite)
            formatted_pops = []
            for p in population:
                ind = [f'{x:.3f}' for x in p.rnvec]
                formatted_pop = '   '.join(ind)
                formatted_pops.append(formatted_pop)
            formatted_pops = '\n'.join(formatted_pops)
            writer[0].add_text(f'Population/Pop_Rep{rep}', formatted_pops, ite)
            formatted_bestinds = []
            for p in gbest:
                ind = [f'{x:.3f}' for x in p]
                formatted_bestind = '   '.join(ind)
                formatted_bestinds.append(formatted_bestind)
            formatted_bestinds = '\n'.join(formatted_bestinds)
            writer[0].add_text(f'Population/Gbest_Rep{rep}', formatted_bestinds, ite)
            for i in range(no_of_tasks):
                if noImpove[i] > 0:
                    converge[i] += 1
                else:
                    converge[i] = 0
            ite += 1
            recorder[rep]['Pops'].append([population[i].rnvec for i in range(pop)])
            recorder[rep]['Fval'].append([population[i].cost for i in range(pop)])
            recorder[rep]['SkillFactor'].append([population[i].task_id for i in range(pop)])
            factorial_cost = np.zeros(pop)
        bestPop[rep, :] = population           
    
    data_MFPSO = {
        'EvBestFitness': EvBestFitness,
        'bestInd_data': bestInd_data,
        'TotalEvaluations': TotalEvaluations
    }
    with open('./MFPSO_data.pkl', 'wb') as f:
        pickle.dump(recorder, f)
    skillFactor = [recorder[rep]['SkillFactor'][-1] for rep in range(reps)]
    return bestPop, skillFactor, bestInd_data