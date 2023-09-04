import torch
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, accuracy_score

def valued_pairs(target, pred):
    assert len(target) == len(pred)
    if type(target)==list:
        target=np.array(target)
    if type(pred)==list:
        pred = np.array(pred)
    return (target != 5) & (pred != 5)

# def acc_stages(target, pred):
#     idx = valued_pairs(target, pred)
#     return accuracy_score(target[idx], pred[idx])
# def f1_stages(target, pred, average = 'macro'):
#     idx = valued_pairs(target, pred)
#     return f1_score(target[idx], pred[idx], average=average)
# def kappa_stages(target, pred):
#     idx = valued_pairs(target, pred)
#     return cohen_kappa_score(target[idx], pred[idx])

### shape of targets and preds: [batch, number]            
def F1_Event(targets:torch.Tensor, preds: torch.Tensor, thresh = .2):
    def _findEvents(x, condition = 'x!=0', return_idx = False):
        i = istart = 0; events = []
        idxtp = torch.where(eval(condition))[0]
        idxtpdiff = torch.zeros_like(idxtp)
        idxtpdiff[0:-1] = torch.diff(idxtp,) == 1
        while i < len(idxtpdiff):
            if not idxtpdiff[i]:
                events.append([idxtp[istart], idxtp[i]])
                istart = i+1
            i+=1
        if not return_idx:
            events = [x[e[0]:e[1]+1] for e in events]
        return np.array(events)
    def _auc(t, p):
        idx = sorted(t.tolist() + p.tolist())
        return (idx[2]-idx[1]) / (idx[3]-idx[0])

    TP = FN = FP = 0
    
    assert preds.shape == targets.shape
    true = targets.view(-1) * 2
    pred = preds.view(-1)
    tp = true + pred

    # split tp to events
    events = _findEvents(tp, 'x!=0')
    events = [e for e in events if len(e)>int(.3*128)]
    
    for ev in events:
        ts = _findEvents(ev, 'x>=2', return_idx=True)
        ps = _findEvents(ev, 'x!=2', return_idx=True)
        ts = np.array([t for t in ts if t[1]-t[0] >= int(.3*128)])
        ps = np.array([p for p in ps if p[1]-p[0] >= int(.3*128)])
        FPEV = len(ps)
        for (i, t) in enumerate(ts):
            auc_t = torch.empty(len(ps))
            for (j, p) in enumerate(ps):
                auc_t[j] = _auc(t, p)
            if len(auc_t) == 0 or torch.max(auc_t) < thresh:
                FN += 1
            else:
                deleted_idx = torch.max(auc_t) - auc_t 
                ps = ps[deleted_idx.bool().numpy()]
                FPEV -=1 
                TP += 1
        FP += FPEV 
    P = TP/(TP+FP+1e-5)
    R = TP/(TP+FN+1e-5)
    F1 = 2*P*R/(P+R+1e-5)
    print(TP, FP, FN)
    print(P, R, F1)
    ###      pred
    # true   1    0
    #   1   TP1  FN3
    #   0   FP2  TN-   
    ###
    return F1

###
# target: list(n_subj x n_epoch^)
# pred: list(n_method x n_subj x n_epoch^)
def metrics_psg_stat(target_list:list, pred_list:list):
    for pred in pred_list:
        assert len(pred) == len(target_list)
    n_subj = len(target_list)

    # metrics: ACC KAPPA F1
    acc = np.zeros((len(pred_list), len(target_list))) - 1
    mf1 = np.zeros((len(pred_list), len(target_list))) - 1
    kappa = np.zeros((len(pred_list), len(target_list))) - 1
    for i in range(len(target_list)):
        for j in range(len(pred_list)):
            vidx = valued_pairs(target_list[i], pred_list[j][i])
            acc[j][i] = accuracy_score(target_list[i][vidx], pred_list[j][i][vidx])
            mf1[j][i] = f1_score(target_list[i][vidx], pred_list[j][i][vidx], average='macro')
            kappa[j][i] = cohen_kappa_score(target_list[i][vidx], pred_list[j][i][vidx])
    
    mean_acc = np.mean(acc, axis=1)
    mean_kappa = np.mean(kappa, axis=1)
    mean_mf1 = np.mean(mf1, axis = 1)
    std_acc = np.std(acc, axis=1)
    std_kappa = np.std(kappa, axis=1)
    std_mf1 = np.std(mf1, axis = 1)
    return mean_acc, std_acc, mean_kappa, std_kappa, mean_mf1, std_mf1

###
# target: dict(idx: list(n_epoch^))
# pred: dict(idx: list(n_epoch^))
def metrics_cohort_stat(target_list:dict, pred_list:dict):
    def stat_cohorts(metrics, weights = None):
        matrix = np.array([metrics[name] for name in metrics]).reshape((len(metrics), -1))
        average = np.average(matrix, axis = 0, weights = weights)
        variance = np.average((matrix-average) ** 2, axis = 0, weights = weights)
        return (average, variance ** .5)

    assert len(target_list) == len(pred_list)
    cohort_idx = {
            'SHHS': (0, 8287),
            'CCSHS': (8288, 8802),
            'SOF': (8803, 9251),
            'CFS': (9252, 9973),
            'MROS': (9974, 13859),
            'MESA': (13860, 15893),
            'HPAP':(15894, 16138),
            'ABC':(16139, 16269),
            'NCHSDB':(16270, 17251),
            'MASS13':(17252, 17366),
            'HMC':(17367, 17520),
            'SSC':(17521, 18288),
            'CNC':(18289, 18365),
            'PHY':(18366, 19358),
            'DOD':(19359, 19439),
            'DHC':(19440, 19521),
            'DCSM':(19522, 19776),
            # 'SHHS1ex':(19777, 19902),
            'WSC':(19903, 22436),
            'ISRC':(22437, 22505)}
    hyp_data = {c_name:[torch.IntTensor([]), torch.IntTensor([])] for c_name in cohort_idx}
    for idx in target_list:
        for c_name in cohort_idx:
            if cohort_idx[c_name][0] <= int(idx) <= cohort_idx[c_name][1]:
                hyp_data[c_name][0] = torch.cat((hyp_data[c_name][0], target_list[idx]))
                hyp_data[c_name][1] = torch.cat((hyp_data[c_name][1], pred_list[idx]))
                break
    # calculate metrics
    n_record = {}
    metrics_global = {}
    f1_stage = {}
    for c_name in hyp_data:
        if len(hyp_data[c_name][0]):
            n_record[c_name] = len(hyp_data[c_name][0])

            _metrics = metrics_psg_stat([hyp_data[c_name][0]], [[hyp_data[c_name][1]]])
            metrics_global[c_name] = np.array(_metrics)[[0,2,4]].squeeze(1)

            vidx = valued_pairs(hyp_data[c_name][0], hyp_data[c_name][1])
            f1_stage[c_name] = f1_score(hyp_data[c_name][0][vidx], hyp_data[c_name][1][vidx], average = None)
    
    weights = [n_record[name] for name in metrics_global]
    metrics_global['mean_weighted'], metrics_global['std_weighted'] = \
                        stat_cohorts(metrics_global, weights)
    f1_stage['mean_weighted'], f1_stage['std_weighted'] = \
                        stat_cohorts(f1_stage, weights)
    return metrics_global, f1_stage

if __name__ == '__main__':
    pass