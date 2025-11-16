import numpy as np
import time
import joblib
import pickle
import torch
from copy import deepcopy
from EA.Particle import Particle
from scipy.optimize import minimize
from tqdm import tqdm
from core.model import G_D, Proxy, DomainClassifier
from core.trainer_GD import train_GD, train_DCls, train_proxy

def evaluateProxy(args, GD, P, pops):
    device = args.device
    all_rnvec = torch.FloatTensor([p.rnvec for p in pops]).to(device)
    all_sf = torch.LongTensor([p.skill_factor for p in pops]).unsqueeze(1).to(device)
    
    GD(all_rnvec, y_fake=all_sf, train_G=True)
    tp_feat = GD.get_G_feat()
    fval = P(tp_feat).detach().cpu().numpy()
    for i, p in enumerate(pops):
        p.factorial_costs[:] = np.inf
        p.factorial_costs[p.skill_factor] = fval[i]
    return pops

def GDPMFPSO(Tasks, options, params, writer):
    pop = options['popsize']
    gen = options['maxgen']
    rmp = options['rmp']
    reps = options['reps']
    args = params['args']
    cfg = args.GD_config

    GD_update = args.GD_update
    t_num = len(params['groups'])
    in_channel = params['Lb'].shape[0]
    GD = G_D(t_num, in_channel, cfg['G_model']['f_num'], cfg['D_model']['f_num'], D_activation='relu',D_norm=None,G_activation='relu',G_norm='layer')
    DCls = DomainClassifier(cfg['G_model']['f_num'], t_num, activation='relu',norm=None)
    P = Proxy(cfg['G_model']['f_num'], activation='relu', norm=None)
    device = args.device
    GD = GD.to(device)
    DCls = DCls.to(device)
    P = P.to(device)
    GD.register_hook()
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
    
    options = {'disp': False, 'maxiter': 5}  # 个体学习的设置
    recorder = [{'Pops':[],'Fval':[],'SkillFactor':[]} for _ in range(reps)]
    fnceval_calls = np.zeros(reps)
    calls_per_individual = np.zeros(pop)
    EvBestFitness = np.zeros((no_of_tasks * reps, gen))  # 到目前为止找到的最佳适应度
    TotalEvaluations = np.zeros((reps, gen))  # 到目前为止的任务评估总数
    bestobj = np.inf * np.ones(no_of_tasks) # 到目前为止找到的任务最优目标值
    bestPop = np.zeros((reps, no_of_tasks), dtype=object)
    # bestPop = np.zeros((reps, pop), dtype=object)

    data = []
    task_id = []
    fval = []
    with joblib.Parallel(n_jobs=6, backend='threading') as parallel:
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
                gbest = np.array([population[0].rnvec for _ in range(no_of_tasks)])
                EvBestFitness[i + 2 * (rep - 1), 0] = bestobj[i]
                bestInd_data = np.array([population[0] for _ in range(no_of_tasks)])
            
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

            pass    
            data.extend([p.rnvec for p in population])
            task_id.extend([p.skill_factor for p in population])
            fval.extend([p.factorial_costs[p.skill_factor] for p in population])
            x = torch.FloatTensor(np.array(data)).to(device)
            y = torch.LongTensor(task_id).to(device).unsqueeze(1)
            GD.train()
            train_GD(params, GD, x, y, writer[0])
            GD.eval()
            GD(x=x, y_fake=y)
            x = GD.get_G_feat()
            train_DCls(params, DCls, x, y, writer[0]) 
            y = torch.FloatTensor(fval).to(device).unsqueeze(1)
            train_proxy(params, P, x, y, writer[0])

            ite = 1
            noImpove = 0
            converge = 0
            restart_flag = 0
            while ite <= gen:

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
                # writer[0].add_text(f'Population/Pop_Rep{rep}', formatted_pops, ite)
                formatted_bestinds = []
                for p in gbest:
                    ind = [f'{x:.3f}' for x in p]
                    formatted_bestind = '   '.join(ind)
                    formatted_bestinds.append(formatted_bestind)
                formatted_bestinds = '\n'.join(formatted_bestinds)
                # writer[0].add_text(f'Population/Gbest_Rep{rep}', formatted_bestinds, ite)
                

                st = time.time()
                w1 = wmax - (wmax - wmin) * ite / 1000
                
                # if converge >= 20:
                #     break
                if ite % 10 == 0 and noImpove >= 20:
                    # 重启
                    for i in range(pop):
                        population[i].velocityUpdate(gbest, rmp, w11, c11, c22, c33)
                    restart_flag = 1
                else:# 用知识迁移或者搜索更新
                    for i in range(pop):
                        if np.random.rand() < 0.5:
                            all_pb = [population[i].pbest for i in range(pop)]
                            all_sk = [population[i].skill_factor for i in range(pop)]
                            population[i].velocityUpdateGD(GD, DCls, all_pb, all_sk, gbest, rmp, w11, c11, c22, c33, writer[0])
                        else:
                            population[i].velocityUpdate(gbest, rmp, w1, c1, c2, c3)
                
                for i in range(pop):
                    population[i].positionUpdate()
                for i in range(pop):
                    population[i].pbestUpdate()              
                
                if (ite % GD_update == 0) or (restart_flag == 1):#用原始评估函数评估解并收集数据
                    restart_flag = 0
                    results = parallel(joblib.delayed(population[i].evaluate)(Tasks, no_of_tasks, params)
                    for i in range(pop))

                    for r in results:
                        fnceval_calls[rep] += r[1]
                        population[r[2]] = r[0]  

                    data.extend([p.rnvec for p in population])
                    task_id.extend([p.skill_factor for p in population])
                    fval.extend([p.factorial_costs[p.skill_factor] for p in population])
                    _, uidx = np.unique(np.hstack((np.array(task_id).reshape(-1,1), np.array(data))), axis=0, return_index=True)
                    data = [data[i] for i in uidx]
                    task_id = [task_id[i] for i in uidx]
                    fval = [fval[i] for i in uidx]
                else:#用proxy评估
                    pops = evaluateProxy(args, GD, P, population)
                    population = pops
                
                factorial_cost = np.zeros(pop)
                pb_pop = []
                pb_task_id = []
                for i in range(no_of_tasks):
                    for j in range(pop):
                        factorial_cost[j] = population[j].factorial_costs[i]
                    sorted_indices = np.argsort(factorial_cost)
                    population = [population[i] for i in sorted_indices]
                    for j in range(pop):
                        population[j].factorial_ranks[i] = j
                    if population[0].factorial_costs[i] <= bestobj[i]:
                        pb_pop.append(population[0])
                        pb_task_id.append(i)
                        # bestobj[i] = population[0].factorial_costs[i]                   
                        # gbest[i, :] = population[0].rnvec
                        # bestInd_data[i, ] = population[0]
                        noImpove = 0
                    else:
                        noImpove += 1
                    EvBestFitness[i + 2 * (rep - 1), ite-1] = bestobj[i]
                
                if ite % GD_update != 0:
                    results = parallel(joblib.delayed(p.evaluate)(Tasks, no_of_tasks, params)
                    for p in pb_pop)
                    pops = [r[0] for r in results]
                    pb_pop_pid = np.array([p.p_id for p in pb_pop])
                    sorted_indices = [np.where(p.p_id==pb_pop_pid)[0] for p in pops]
                    pb_pop = [pops[int(i)] for i in sorted_indices]
                for i, t in enumerate(pb_task_id):
                    if pb_pop[i].factorial_costs[t] <= bestobj[t]:
                        bestobj[t] = pb_pop[i].factorial_costs[t]                   
                        gbest[t, :] = pb_pop[i].rnvec
                        bestInd_data[t, ] = pb_pop[i]
                            
                if ite % GD_update == 0:#更新GD和proxy
                    x = torch.FloatTensor(np.array(data)).to(device)
                    y = torch.LongTensor(task_id).to(device).unsqueeze(1)
                    GD.train()
                    train_GD(params, GD, x, y, writer[0])
                    GD.eval()
                    GD(x=x, y_fake=y)
                    x = GD.get_G_feat()
                    train_DCls(params, DCls, x, y, writer[0]) 
                    y = torch.FloatTensor(fval).to(device).unsqueeze(1)
                    train_proxy(params, P, x, y, writer[0])
                    pass
                
                if noImpove > 0:
                    converge += 1
                else:
                    converge = 0
                ite += 1
                recorder[rep]['Pops'].append([population[i].rnvec for i in range(pop)])
                recorder[rep]['Fval'].append([population[i].factorial_costs[population[i].skill_factor] for i in range(pop)])
                recorder[rep]['SkillFactor'].append([population[i].skill_factor for i in range(pop)])
                factorial_cost = np.zeros(pop)

            bestPop[rep, :] = bestInd_data      
            # bestPop[rep, :] = population           
    
    data_MFPSO = {
        'EvBestFitness': EvBestFitness,
        'bestInd_data': bestInd_data,
        'TotalEvaluations': TotalEvaluations
    }
    GD.remove_hook()
    with open('./MFPSO_data.pkl', 'wb') as f:
        pickle.dump(recorder, f)
    return bestPop