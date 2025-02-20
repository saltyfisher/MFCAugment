import numpy as np
from scipy.optimize import minimize

def fnceval(Task, rnvec, p_il, options):
    d = Task.dims
    nvars = rnvec[:d]
    minrange = Task.Lb[:d]
    maxrange = Task.Ub[:d]
    y = maxrange - minrange
    vars = y * nvars + minrange  # decoding
    
    if np.random.rand() <= p_il:
        result = minimize(Task.fnc, vars, method='BFGS', options=options)
        x = result.x
        objective = result.fun
        exitflag = result.success
        output = result.nfev
        
        nvars = (x - minrange) / y
        m_nvars = nvars.copy()
        m_nvars[m_nvars < 0] = 0
        m_nvars[m_nvars > 1] = 1
        
        if not np.array_equal(m_nvars, nvars):
            nvars = m_nvars
            x = y * nvars + minrange
            objective = Task.fnc(x)
        
        rnvec[:d] = nvars
        funcCount = output
    else:
        x = vars
        objective = Task.fnc(x)
        funcCount = 1
    
    return objective, rnvec, funcCount