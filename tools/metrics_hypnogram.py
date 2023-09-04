import scipy.optimize as optimize
import torch
import numpy as np
from scipy import stats
from scipy.special import kl_div

class HypVariables():
    def __init__(self, *args) -> None:
        (self.TST,   #0
        self.SOL,   #1
        self.LREM,  #2
        self.SE,    #3
        self.WASO,  #4
        self.N1,    #5
        self.N2,    #6
        self.N3,    #7
        self.R,     #8
        self.SSI,    #9
        self.NSI,   #10
        self.RSI,   #11, #12,
        self.Markov) = args
        self.variables = args[:-1]
    def len():
        return 12 + 1 # KL
    def __len__(self):
        len()

def metricsOneHypno(hyp) -> HypVariables:
    def target_func_exp(x, theta):
        return np.exp(-theta * x)

    def survival_analysis(NREM_idx):
        i = 0; NREM_list = []
        istart = 0
        for i in range(len(NREM_idx) - 1):
            if (not NREM_idx[i]) and NREM_idx[i+1]:   # (0, 1)
                istart = i+1
            elif NREM_idx[i] and (not NREM_idx[i+1]): # (1, 0)
                NREM_list.append([istart, i])
        # NREM_list: [n, 2]
    
        # NREM_dur = [i[1] - i[0] + 1 for i in NREM_list if i[1] != i[0]]   # ignore fragments < 1min
        NREM_dur = [i[1] - i[0] + 1 for i in NREM_list]                   # do not ignore fragments < 1min
        if len(NREM_dur) == 0:
            return 1.
        
        NREM_dur = np.array(NREM_dur)
        survival_rate = np.zeros(np.max(NREM_dur) + 1)
        for i in range(np.max(NREM_dur) + 1):
            survival_rate[i] = (NREM_dur>i).sum()
        survival_rate = survival_rate / survival_rate[0]
        if len(survival_rate) == 2:
            theta = 1.
        else:
            try:
                theta, _ = optimize.curve_fit(target_func_exp, xdata = np.arange(len(survival_rate)) / 2, ydata = survival_rate, p0=[0])
            except:
                theta = 1.
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(NSrate)),NSrate)
        # plt.plot(np.arange(len(NSrate)),target_func_exp(np.arange(len(NSrate)), theta))
        # plt.show()
        return min(float(theta), 1.)

    # 0~5: Wake, N1, N2, N3, REM, Unknown
    if type(hyp) == torch.Tensor:
        hyp = hyp.numpy()
    assert 6 not in hyp
    hyp = hyp.astype(int)

    # sleep architecture
    hyp_known = hyp[hyp!=5]
    N = len(hyp_known)
    [Widx, N1idx, N2idx, N3idx, Ridx] = [hyp_known == i for i in range(5)]
    _W = sum(Widx) / N * 100        # %
    _R = sum(Ridx) / N * 100        # %
    _N1 = sum(N1idx) / N * 100      # %
    _N2 = sum(N2idx) / N * 100      # %
    _N3 = sum(N3idx) / N * 100      # %

    # stage shift
    _markov = np.zeros((5,5))
    for i in range(N-1):
        _markov[hyp_known[i],hyp_known[i+1]] += 1
    _SSI = (N-1 - sum(_markov[i,i] for i in range(5))) / (N / 60 / 2)
    
    # REM/NREM survival
    N_idx = N1idx + N2idx + N3idx
    N_idx = np.concatenate([N_idx, [0]])
    _NSI = survival_analysis(N_idx)
    R_idx = np.concatenate([Ridx, [0]])
    _RSI = survival_analysis(R_idx)

    # sleep continuity (with Unknown stages)
    N = len(hyp)
    __tmp_stage = 0 # assign Unknown as their preceding Known stage
    for i in range(N):
        __tmp_stage = __tmp_stage if hyp[i] == 5 else hyp[i]
        hyp[i] = __tmp_stage
    [Widx, N1idx, N2idx, N3idx, Ridx] = [hyp == i for i in range(5)]
    sleepIdx = N1idx + N2idx + N3idx + Ridx
    sleep_idx = np.where(sleepIdx != 0)[0]
    if len(sleep_idx) > 0:
        sleep_start = sleep_idx[0] 
        sleep_end = sleep_idx[-1]

        _TST = len(sleep_idx) * 0.5                                 # minute
        _SOL = sleep_start * 0.5                                    # minute

        R_idx = np.where(Ridx!=0)[0]
        R_idx = [sleep_end] if len(R_idx) == 0 else R_idx           # no REM sleep
        _ROL = R_idx[0] * .5 - _SOL                                 # minute
        _SE = _TST / (N/2) * 100                                    # %
        _WASO = (hyp[sleep_start:] == 0).sum() * .5                 # minute
    else:
        _WASO = 0                                                   # NULL
        _SOL = _ROL = N-1                                           # end of psg
        _TST = _SE = 0                                              # 0

    return HypVariables(_TST, _SOL, _ROL, _SE, _WASO, _N1, _N2, _N3, _R,
                         _SSI, _NSI, _RSI, _markov)

# cal hyp metrics in ONE psg.
# input gt : list() 
# input preds = [pred0, pred/1...] : [list(), list()...]
def metricsHypsOnePSG(gt, preds):
    for pred in preds:
        assert len(pred) == len(gt)

    # print('Calculating metrics for %d groups...'%(len(preds) + 1))
    table_vars = np.zeros((len(preds) + 1, HypVariables.len())) - 1     # (with KL)
    markov = np.zeros((len(preds) + 1,5,5))

    # t&p
    tV = metricsOneHypno(gt)
    table_vars[0][:-1] = tV.variables
    markov[0] += tV.Markov
    mkv_cvt = lambda x: (x / (np.sum(x,1).reshape([-1,1]) + 1e-8)).flatten() + 1e-8
    for j in range(len(preds)):
        pV = metricsOneHypno(preds[j])
        table_vars[j+1,:-1] = pV.variables
        table_vars[j+1,-1] = kl_div(mkv_cvt(tV.Markov), mkv_cvt(pV.Markov)).sum()
        markov[j+1] += pV.Markov
    # assert -1 not in table_vars
    return table_vars, markov

# cal distribution and MAE(MSE)
# target_list: list(), shape = [n_subj, n_epoch^] 
# pred_list: [pred0, pred1... ], shape = [n_pred, n_subj, n_epoch*]
def statDistributions(target_list:list, pred_list:list):
    for pred in pred_list:
        assert len(pred) == len(target_list)
    print('In statistics...')
    n_subj = len(target_list)

    # get vars&markovs
    vars = np.zeros((n_subj, 1+len(pred_list), HypVariables.len())) - 1
    markov_global = np.zeros((1+len(pred_list),5,5))
    for i in range(n_subj):
        t,m = metricsHypsOnePSG(target_list[i], [pred[i] for pred in pred_list])
        vars[i] = t
        markov_global += m

    # value
    p_metrics = []
    mean_metrics = np.mean(vars, axis = 0, where=vars>=0)
    std_metrics = np.std(vars, axis = 0, where=vars>=0)
    for i in range(1, vars.shape[1]):
        _p_metrics = []
        for j in range(HypVariables.len() - 1):
            valued = (vars[:,i,j]>0) & (vars[:,0,j]>0)
            if stats.levene(vars[:,i,j], vars[:,0,j])[1] > .05:
                _p_metrics.append([float(stats.ttest_rel(vars[valued,i,j], vars[valued,0,j])[1]), 'ttest_rel'])
                # _p_metrics.append([float(stats.f_oneway(vars[valued,i,j], vars[valued,0,j])[1]), 'anova'])
            else:
                _p_metrics.append([float(stats.wilcoxon(vars[valued,i,j], vars[valued,0,j])[1]), 'wilcoxon'])
                # _p_metrics.append([float(stats.kruskal(vars[valued,i,j], vars[valued,0,j])[1]), 'kruskal'])
        p_metrics.append(_p_metrics)

    # residual
    vars_res = np.abs(vars[:,:,:] - np.expand_dims(vars[:,0,:],1))
    p_res = []  # n_pred > 1
    mean_res = np.mean(vars_res, axis = 0, where=vars_res>=0)
    # mean_res = np.mean(vars_res, axis = 0, where = ~np.any((vars_res>=0)==0, axis = 1))   ???
    std_res = np.std(vars_res, axis = 0, where=vars_res>=0)
    for i in range(1, vars_res.shape[1]):
        _p_res = []
        for j in range(HypVariables.len()):
            valued = (vars_res[:,i,j]>0) & (vars_res[:,1,j]>0)
            if valued.sum() == 0:
                _p_res.append(-1)
                continue
            if stats.levene(vars_res[:,i,j], vars_res[:,1,j])[1] > .05:
                _p_res.append([float(stats.ttest_rel(vars_res[valued,i,j], vars_res[valued,0,j])[1]), 'ttest_rel'])
                # _p_res.append([float(stats.f_oneway(vars_res[valued,i,j], vars_res[valued,1,j])[1]), 'anova'])
            else:
                _p_res.append([float(stats.wilcoxon(vars_res[valued,i,j], vars_res[valued,0,j])[1]), 'wilcoxon'])
                # _p_res.append([float(stats.kruskal(vars_res[valued,i,j], vars_res[valued,1,j])[1]), 'kruskal'])
        p_res.append(_p_res)

    # markov
    for i in range(markov_global.shape[0]):
        markov_global[i] = markov_global[i] / (np.sum(markov_global[i],1).reshape([-1,1]) + 1e-8)
    return mean_metrics, std_metrics, p_metrics, mean_res, std_res, p_res, markov_global

if __name__ == '__main__':
    pass