import warnings
import numpy as np
import torch,os,pickle
from os.path import join
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from torch.nn import DataParallel as DP
from tools import *
from tools.dataset_feature import XY_dataset_feat

np.set_printoptions(suppress=True) 
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class FeatRegularizer():
    def __init__(self, cuda = False) -> None:
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.hypLen = -1

    def feat_to_batch(self, feat, n_epoch = 1260):
        # feature shape = [bs, n_ep, 6]
        L, *C = feat.shape
        self.hypLen = L
        if L < n_epoch:
            batches = torch.cat([feat, torch.zeros([n_epoch - L, *C]).to(self.device)], dim=0)
            return batches.unsqueeze(0)
        else:
            overlap = 5 # 80% overlap
            step = n_epoch // overlap 
            batches = torch.zeros([(L - n_epoch + step - 1) // step + 1, n_epoch, *C]).to(self.device)
            i_forward = -1
            while (i_forward + overlap) * step < feat.shape[0]:
                i_forward += 1 
                batch = feat[i_forward * step:(i_forward + overlap) * step]
                batches[i_forward][:batch.shape[0]] = batch
            return batches

    def batch_to_hypno(self, preds): 
        n_batch, n_epoch, *dims = preds.shape
        if n_batch == 1:
            pred_hyp = preds[0]
        else:
            overlap = 5 # 80% overlap
            step = n_epoch // overlap 
            pred_hyp = torch.zeros([(n_batch - 1) * step + n_epoch, *dims]).to(self.device)
            for i in range(n_batch):
                pred_hyp[i*step:i*step+n_epoch] += torch.softmax(preds[i], 1)
        # pred_hyp = np.argmax(pred_hyp, axis = 1)
        return pred_hyp[:self.hypLen]


def tester_hyp(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    model = torch.load(cfg.best_hyp_ckp)['net'].cuda()
    if type(model) == DP:
        model = model.module
    if cfg.eval_parallel:
        model = DP(model)

    testSet = XY_dataset_feat(cfg, tvt=cfg.tvt)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = 1, persistent_workers=False,
                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)
    model.eval()

    ### init vars
    psgs_pred_hyp = {}
    psgs_target = {}
    Regulr = FeatRegularizer(cuda = True)

    with torch.no_grad():
        tq = tqdm(testLoader, desc= 'Test', ncols=80, ascii=True)
        for idx, (feature, target, psg) in enumerate(tq):
            feature = torch.squeeze(feature, 0).cuda(non_blocking = True) # TODO batch > 1
            batches = Regulr.feat_to_batch(feature)

            pred_hyp = torch.empty([0, cfg.PSG_EPOCH, cfg.SWIN.IN_CHANS]).cuda()
            for batch in batches:
                # forward TODO batch > 1
                pred, _ = model(batch.unsqueeze(0))
                pred_hyp = torch.cat([pred_hyp, pred])
            
            pred_hyp = Regulr.batch_to_hypno(pred_hyp)
            target = target.squeeze(0)
            torch.cuda.empty_cache()
            
            # record
            pred_hyp = torch.argmax(pred_hyp, -1).cpu()
            psgs_pred_hyp[int(psg)] = pred_hyp
            psgs_target[int(psg)] = target

    # BUGFIX: label unknown
    for i in psgs_target:
        if torch.count_nonzero(psgs_target[i] - 5) == 0:
            psgs_target[i][0] = 0

    # performance report
    report = 'Global...\n'
    y_true, y_pred = [], []
    for s in psgs_target:
        y_true += psgs_target[s]
        y_pred += psgs_pred_hyp[s]
    vidx = valued_pairs(y_true, y_pred)
    y_true = np.array(y_true)[vidx]
    y_pred = np.array(y_pred)[vidx]
    report += str(confusion_matrix(y_true, y_pred)) + '\n'
    report += classification_report(y_true, y_pred)
    report += '\nAcc=%.4f'%(accuracy_score(y_true, y_pred))
    report += '\nKappa=%.4f'%(cohen_kappa_score(y_true, y_pred))
    report += '\nMF1=%.4f'%(f1_score(y_true, y_pred, average = 'macro'))

    report += '\n\nby-Cohort...(cohort name, (acc, kappa, mf1), (f1 x 5))\n'
    m_cohort, f1_cohort = metrics_cohort_stat(psgs_target, psgs_pred_hyp)
    for name in m_cohort:
        report += '%s:%s,%s\n'%(name, str(m_cohort[name]), str(f1_cohort[name]))

    report += '\nby-PSG...(acc, kappa, mf1)\n'
    target_list=[psgs_target[i] for i in psgs_target]
    pred_list=[[psgs_pred_hyp[i] for i in psgs_pred_hyp]]
    m_psg = metrics_psg_stat(target_list, pred_list)
    report += '%.4f,%.4f\n%.4f,%.4f\n%.4f,%.4f'%m_psg

    report += '\n\nby-Hypnogram...(HypVariables of TARGET & PRED)\n'
    m_hyp = statDistributions(target_list, pred_list)
    report += 'mean:\n%s\nstd:\n%s\np:\n%s\nmean_res:\n%s\nstd_res:\n%s\np_res\n\%s\nmarkov:\n%s'%(m_hyp)
            
    # save
    print('saving ...')
    assert len(psgs_pred_hyp) == len(psgs_target)
    saveResult = join('./experiments/%s/hyp_test_result.pkl'%(cfg.experiment))
    with open(saveResult, 'wb') as f:
        pickle.dump({'pred':psgs_pred_hyp,'target':psgs_target}, f)
    with open(saveResult.replace('hyp_test_result.pkl','hyp_test_report.txt'), 'w') as f:
        f.write(report)
    print('done!')
    return psgs_pred_hyp, psgs_target


if __name__ == "__main__":
    pass