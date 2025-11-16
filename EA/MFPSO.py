import numpy as np
import time
from scipy.optimize import minimize
from EA.Particle import Particle
from tqdm import tqdm
import joblib
import pickle
def MFPSO(Tasks, options, params, writer):
    pop = options['popsize']
    gen = options['maxgen']
    rmp = options['rmp']
    reps = options['reps']
    if pop % 2 != 0:
        pop += 1
    
    no_of_tasks = len(Tasks)
    if no_of_tasks <= 1:
        raise ValueError('At least 2 tasks required for MFEA')
    
    wmax = 0.9  # 惯性权重
    wmin = 0.4  # 惯性权重

    c1 = 0.2
    c2 = 0.2
    c3 = 0.2
    w11 = 1000
    c11 = 1000
    c22 = 1000
    c33 = 1000

    D = np.zeros(no_of_tasks)
    for i in range(no_of_tasks):
        D[i] = Tasks[i].dims
    D_multitask = int(np.max(D))
    
    options = {'disp': False, 'maxiter': 2}  # 个体学习的设置
    recorder = [{'Pops':[],'Fval':[],'SkillFactor':[]} for _ in range(reps)]
    fnceval_calls = np.zeros(reps)
    calls_per_individual = np.zeros(pop)
    EvBestFitness = np.zeros((no_of_tasks * reps, gen))  # 到目前为止找到的最佳适应度
    TotalEvaluations = np.zeros((reps, gen))  # 到目前为止的任务评估总数
    bestobj = np.inf * np.ones(no_of_tasks) # 到目前为止找到的任务最优目标值
    bestPop = np.zeros((reps, pop), dtype=object)
    bestInd_data = np.zeros((reps, no_of_tasks), dtype=object)
    with joblib.Parallel(n_jobs=1, backend='threading') as parallel:
        for rep in range(reps):
            population = [Particle(D_multitask, no_of_tasks, i) for i in range(pop)]
            
            st = time.time()
            results = parallel(joblib.delayed(population[i].evaluate)(Tasks, no_of_tasks, params)
                for i in range(pop))
            # for i in range(pop):
            #     calls_per_individual[i] = population[i].evaluate(Tasks, no_of_tasks, params)
            for r in results:
                fnceval_calls[rep] += r[1]
                population[r[2]] = r[0]
            TotalEvaluations[rep, 0] = fnceval_calls[rep]
            print(f'Initializing:{time.time()-st:.2f}')
            recorder[rep]['Pops'].append([population[i].rnvec for i in range(pop)])
            recorder[rep]['Fval'].append([population[i].factorial_costs[population[i].skill_factor] for i in range(pop)])
            recorder[rep]['SkillFactor'].append([population[i].skill_factor for i in range(pop)])
            factorial_cost = np.zeros(pop)
            for i in range(no_of_tasks):      
                for j in range(pop):
                    factorial_cost[j] = population[j].factorial_costs[i]
                sorted_indices = np.argsort(factorial_cost)
                population = [population[i] for i in sorted_indices]  # 根据当前任务的factorial_cost重新排序种群
                for j in range(pop):
                    population[j].factorial_ranks[i] = j
                bestobj[i] = population[0].factorial_costs[i]
                bestInd_data[rep, i] = population[0]
                gbest = np.array([population[0].rnvec for _ in range(no_of_tasks)])
                EvBestFitness[i + 2 * (rep - 1), 0] = bestobj[i]
            
            for i in range(pop):
                min_rank = np.min(population[i].factorial_ranks)
                equivalent_skills = np.where(population[i].factorial_ranks == min_rank)[0]
                if len(equivalent_skills) > 1:  # 如果在多个任务上有最佳适应度，随机选择一个并将其factorial_costs设置为inf
                    population[i].skill_factor = equivalent_skills[np.random.randint(len(equivalent_skills))]
                    tmp = population[i].factorial_costs[population[i].skill_factor]
                    population[i].factorial_costs = np.inf * np.ones(no_of_tasks)
                    population[i].factorial_costs[population[i].skill_factor] = tmp
                    population[i].pbestFitness = tmp
                else:  # 否则，只需设置skill_factor并将其factorial_costs设置为inf
                    population[i].skill_factor = np.argmin(population[i].factorial_ranks)
                    tmp = population[i].factorial_costs[population[i].skill_factor]
                    population[i].factorial_costs = np.inf * np.ones(no_of_tasks)
                    population[i].factorial_costs[population[i].skill_factor] = tmp
                    population[i].pbestFitness = tmp
            
            ite = 1
            noImpove = 0
            converge = 0
            while ite <= gen:
                st = time.time()
                w1 = wmax - (wmax - wmin) * ite / 1000
                
                if converge >= 30:
                    break
                if ite % 10 == 0 and noImpove >= 20:
                    # 重启
                    for i in range(pop):
                        population[i].velocityUpdate(gbest, rmp, w11, c11, c22, c33)
                else:            
                    for i in range(pop):
                        population[i].velocityUpdate(gbest, rmp, w1, c1, c2, c3)
                
                for i in range(pop):
                    population[i].positionUpdate()
                for i in range(pop):
                    population[i].pbestUpdate()
                
                results = parallel(joblib.delayed(population[i].evaluate)(Tasks, no_of_tasks, params)
                for i in range(pop))
                # st = time.time()
                # for i in range(pop):
                #     calls_per_individual[i] = population[i].evaluate(Tasks, no_of_tasks, params)
                # print(time.time()-st)
                for r in results:
                    fnceval_calls[rep] += r[1]
                    population[r[2]] = r[0]
                # for i in range(pop):            
                #     calls_per_individual[i] = population[i].evaluate(Tasks, no_of_tasks, params)           
                # fnceval_calls[rep] += np.sum(calls_per_individual)                   
                
                factorial_cost = np.zeros(pop)
                for i in range(no_of_tasks):
                    for j in range(pop):
                        factorial_cost[j] = population[j].factorial_costs[i]
                    sorted_indices = np.argsort(factorial_cost)
                    population = [population[i] for i in sorted_indices]
                    for j in range(pop):
                        population[j].factorial_ranks[i] = j
                    if population[0].factorial_costs[i] <= bestobj[i]:
                        bestobj[i] = population[0].factorial_costs[i]                   
                        gbest[i, :] = population[0].rnvec
                        bestInd_data[rep, i] = population[0]
                        noImpove = 0
                    else:
                        noImpove += 1
                    EvBestFitness[i + 2 * (rep - 1), ite-1] = bestobj[i]
                bestobj_str = [f'{o:.2f}' for o in bestobj]
                print(f'Time:{time.time()-st:.2f} Generation: {ite} Repeat: {rep} Best Fitness: {bestobj_str}')
                for k in range(no_of_tasks):
                    writer[k].add_scalar(f'Best Fitness/Rep{rep}', bestobj[k], ite)
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
                if noImpove > 0:
                    converge += 1
                else:
                    converge = 0
                ite += 1
                recorder[rep]['Pops'].append([population[i].rnvec for i in range(pop)])
                recorder[rep]['Fval'].append([population[i].factorial_costs[population[i].skill_factor] for i in range(pop)])
                recorder[rep]['SkillFactor'].append([population[i].skill_factor for i in range(pop)])
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