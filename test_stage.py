import warnings
import numpy as np
import torch,os,pickle
from os.path import join
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from torch.nn import DataParallel as DP
from tools import *

np.set_printoptions(suppress=True) 
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def tester_5ep(cfg, withFeature = False):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    resultObj = torch.load(cfg.best_ckp)
    model = resultObj['net'].cuda()
    if type(model) == DP:
        model = model.module
    if cfg.eval_parallel:
        model = DP(model)
        
    myDataset = datasets(cfg)
    testSet = myDataset(cfg, tvt=cfg.tvt)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = cfg.BATCH_SIZE*2, 
                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)
    model.eval()

    ### init vars
    psgs_pred = {}
    psgs_target = {}
    psgs_feature = {}
    ###

    with torch.no_grad():
        tq = tqdm(testLoader, desc= 'Test', ncols=80, ascii=True)
        for i, (data, target, psg) in enumerate(tq):
            inputs = Variable(data.cuda())
            pred, feature = model(inputs)

            # record
            pred = torch.argmax(pred,1).cpu()
            feature = feature.cpu()
            for j in range(len(psg)):
                idx_psg = int(psg[j])
                if idx_psg not in psgs_target:
                    psgs_pred[idx_psg] = torch.empty([0]).int()
                    psgs_target[idx_psg] = torch.empty([0]).int()
                    psgs_feature[idx_psg] = torch.empty([0, cfg.SWIN.EMBED_DIM])
                psgs_pred[idx_psg] = torch.cat([psgs_pred[idx_psg], pred[j:j+1].int()])
                psgs_target[idx_psg] = torch.cat([psgs_target[idx_psg], target[j:j+1].int()])
                if withFeature:
                    psgs_feature[idx_psg] = torch.cat([psgs_feature[idx_psg], feature[j:j+1]])
    
    # performance report
    report = 'Global...\n'
    y_true, y_pred = [], []
    for s in psgs_target:
        y_true += psgs_target[s]
        y_pred += psgs_pred[s]
    vidx = valued_pairs(y_true, y_pred)
    y_true = np.array(y_true)[vidx]
    y_pred = np.array(y_pred)[vidx]
    report += str(confusion_matrix(y_true, y_pred)) + '\n'
    report += classification_report(y_true, y_pred)
    report += '\nAcc=%.4f'%(accuracy_score(y_true, y_pred))
    report += '\nKappa=%.4f'%(cohen_kappa_score(y_true, y_pred))
    report += '\nMF1=%.4f'%(f1_score(y_true, y_pred, average = 'macro'))

    report += '\n\nby-Cohort...(cohort name, (acc, kappa, mf1), (f1 x 5))\n'
    m_cohort, f1_cohort = metrics_cohort_stat(psgs_target, psgs_pred)
    for name in m_cohort:
        report += '%s:%s,%s\n'%(name, str(m_cohort[name]), str(f1_cohort[name]))

    report += '\nby-PSG...(acc, kappa, mf1)\n'
    target_list=[psgs_target[i] for i in psgs_target]
    pred_list=[[psgs_pred[i] for i in psgs_pred]]
    m_psg = metrics_psg_stat(target_list, pred_list)
    report += '%.4f,%.4f\n%.4f,%.4f\n%.4f,%.4f'%m_psg

    report += '\n\nby-Hypnogram...(HypVariables of TARGET & PRED)\n'
    m_hyp = statDistributions(target_list, pred_list)
    report += 'mean:\n%s\nstd:\n%s\np:\n%s\nmean_res:\n%s\nstd_res:\n%s\np_res\n\%s\nmarkov:\n%s'%(m_hyp)
            
    # save
    print('saving ...')
    assert len(psgs_pred) == len(psgs_target) == len(psgs_feature)
    saveResult = join('./experiments/%s/stage_test_result.pkl'%(cfg.experiment))
    with open(saveResult, 'wb') as f:
        pickle.dump({'pred':psgs_pred,'target':psgs_target,'feature':psgs_feature}, f)
    with open(saveResult.replace('stage_test_result.pkl','stage_test_report.txt'), 'w') as f:
        f.write(report)
    print('done!')
    return psgs_pred, psgs_target, psgs_feature


if __name__ == "__main__":
    pass