import warnings
import numpy as np
import torch,os
from os.path import join
from torch.autograd import Variable
from torch.nn import DataParallel as DP
from mne.io import read_raw_edf
np.set_printoptions(suppress=True) 
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
multiprocessing.set_start_method('spawn', True)

import os, sys
sys.path.append(os.getcwd())
from tools import *
from scripts.get_meta_feature import psgMetaFeat

_EPOCH_SEC_SIZE = 30
_stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "UNKNOWN" :5
}
chnNameTable = {
    'Keyue':{'C4':'C4-M1', 'C3':'C3-M2', 'F4':'F4-M1', 'F3':'F3-M2',
            'E1':'E1-M2', 'E2':'E2-M2',
            'O2':'O2-M1', 'O1':'O1-M2', 'EMG':'ChinL', 'EMGref':'ChinR'},
    'Xijing':{'C4':'EEG C4-REF', 'C3':'EEG C3-REF', 'F4':'EEG F4-REF', 'F3':'EEG F3-REF',
            'E1':'EOG1', 'E2':'EOG2',
            'O2':'EEG O2-REF', 'O1':'EEG O1-REF', 'EMG':'EMG'},
    'South':{'EMG':'Chin 1', 'EMGref':'Chin 3'
    }
}

def _getExpectedChnNames(chnNameTable, rawChns):
    chnNames = {}
    expect = ['F3','F4','C3','C4','O1','O2','E1','E2','EMG','M1','M2','EMGref']
    for c in expect:
        found = False
        names = chnNameTable[c] if c in chnNameTable else []
        names = [names] if type(names) is str else list(names)
        names.append(c) # default channel name
        for name in names:
            for rcn in rawChns:
                if name.upper() == rcn.upper():
                    chnNames[c] = rcn;found = True;break
            if found:break
    # del exist ref
    for ref in ('M1', 'M2'):
        if ref in chnNames:
            for c in ('F3','F4','C3','C4','O1','O2','E1','E2'):
                if (c in chnNames) and (chnNames[ref] in chnNames[c]):
                    del chnNames[ref]
                    break
    # replace alternative channel if ["F4, C4, O2, EMG"] not exist. BUG: replace ref
    if ('F4' not in chnNames) and ('F3' in chnNames):
        chnNames['F4'] = chnNames['F3']
        del chnNames['F3']
    if ('C4' not in chnNames) and ('C3' in chnNames):
        chnNames['C4'] = chnNames['C3']
        del chnNames['C3']
    if ('O2' not in chnNames) and ('O1' in chnNames):
        chnNames['O2'] = chnNames['O1']
        del chnNames['O1']
    if ('EMG' not in chnNames) and ('EMGref' in chnNames):
        chnNames['EMG'] = chnNames['EMGref']
        del chnNames['EMGref']
    return chnNames

def _checkChnValue(data, sampling_rate = 128):
    n_except = 0
    d = data.flatten()
    n_epoch = len(d) // (_EPOCH_SEC_SIZE * sampling_rate)
    for i in range(n_epoch):
        if not 0.2 < np.diff(np.percentile(d[i*_EPOCH_SEC_SIZE*sampling_rate:(i+1)*_EPOCH_SEC_SIZE*sampling_rate], [25, 75])) < 2e2:
            n_except += 1
    return n_except / n_epoch

def data_generator(edfName = '', sampling_rate = 128, channel = None):
    # load head
    raw = read_raw_edf(edfName, preload=False, stim_channel=None)

    # get raw Sample Freq and Channel Name
    sfreq = raw.info['sfreq']
    resample = False if sfreq == sampling_rate else True
    print('【signal sampling freq】:',sfreq)
    ch_names_exist = _getExpectedChnNames(chnNameTable[cfg.dataset] ,raw.ch_names)
    # ch_names ready!

    # load raw signal
    exclude_channel = raw.ch_names
    for cn in set([ch_names_exist[c] for c in ch_names_exist]):
        if cn is not None: 
            exclude_channel.remove(cn)
    raw = read_raw_edf(edfName, preload=True, stim_channel=None, 
                            exclude=exclude_channel)
    # raw.copy().plot(duration = 30, proj = False, block = True)    
    
    # preprocessing
    X_data = None
    X_order = {'F4':0,'C4':1,'O2':2,'E1':3,'E2':4,'EMG':5}
    ch_name_reverse = {ch_names_exist[c]:c for c in ch_names_exist}
    ch_name_alter = {'F4':'F3','C4':'C3','O2':'O1'}
    # start
    eeg_picks = [ch_names_exist[n] for n in ('F4', 'C4', 'O2', 'M1') if n in ch_names_exist]
    eog_picks = [ch_names_exist[n] for n in ('E1', 'E2', 'M2') if n in ch_names_exist]
    emg_picks = [ch_names_exist[n] for n in ('EMG', 'EMGref') if n in ch_names_exist]
    raw_eeg = raw.copy().pick(eeg_picks)
    raw_eog = raw.copy().pick(eog_picks)
    raw_emg = raw.copy().pick(emg_picks)
    if 'M1' in ch_names_exist:
        raw_eeg.set_eeg_reference([ch_names_exist['M1']])
        raw_eeg = raw_eeg.pick(eeg_picks[:-1])
    if 'M2' in ch_names_exist:
        raw_eog.set_eeg_reference([ch_names_exist['M2']])
        raw_eog = raw_eog.pick(eog_picks[:-1])
    if 'EMGref' in ch_names_exist:
        raw_emg.set_eeg_reference([ch_names_exist['EMGref']])
        raw_emg = raw_emg.pick(emg_picks[:-1])
    for r in (raw_eeg, raw_eog, raw_emg):
        assert len(r.ch_names) > 0
        if r.info['sfreq'] > 120:
            r.notch_filter([50,60])
        if r is not raw_emg:
            r.filter(l_freq = 0.3, h_freq = 35, method='iir')
        else:
            r.filter(l_freq = 10, h_freq = 49, method='iir')
        if resample:
            r.resample(sampling_rate)

        # channel by channel check
        for c in r.ch_names:
            c_data, _ = r[c]    # shape: [1, length]
            ################  Unit: Volt to μV  ################       
            if (r._orig_units[c] in ('µV', 'mV')) or (np.std(c_data) < 1e-3):
                c_data *= 1e6
            #################### Check Unit ####################         
            p = np.percentile(c_data, [5, 25, 75, 95])
            assert 1 < p[2]-p[1] < 1e3 or 5e-2<np.std(c_data)<3e3 or p[3]-p[0] < 1
            ##################### eeg alter ####################
            if (r is raw_eeg) and (ch_name_alter[ch_name_reverse[c]] in ch_names_exist):
                bad_epoch_rate = _checkChnValue(c_data)
                if bad_epoch_rate > 0.1:
                    cname_alter = ch_name_alter[ch_name_reverse[c]]   # F3,C3,O1
                    cname_alter = [ch_names_exist[cname_alter], ch_names_exist['M1']] if 'M1' in ch_names_exist else [ch_names_exist[cname_alter]]
                    r_alter = raw.copy().pick(cname_alter)
                    if 'M1' in ch_names_exist:
                        r_alter.set_eeg_reference([ch_names_exist['M1']])
                        r_alter = r_alter.pick(cname_alter[:-1])
                    if r_alter.info['sfreq'] > 120:
                        r_alter.notch_filter([50,60])
                    r_alter.filter(l_freq = 0.3, h_freq = 35, method='iir')
                    if resample:
                        r_alter.resample(sampling_rate)
                    data_alter, _ = r_alter[cname_alter[0]]
                    if (r_alter._orig_units[cname_alter[0]] in ('µV', 'mV')) or (np.std(data_alter) < 1e-3):
                        data_alter *= 1e6
                    bad_alter_rate = _checkChnValue(data_alter)
                    if bad_alter_rate < bad_epoch_rate:
                        c_data = data_alter
            ####################### Save #######################
            if X_data is None:
                X_data = np.zeros([c_data.shape[1], 6])
            X_data[:, X_order[ch_name_reverse[c]]] = c_data
    # X_data shape: [length, chn]
    n_epochs = len(X_data) // (_EPOCH_SEC_SIZE * sampling_rate)
    X_data_slim = X_data[:int(n_epochs * _EPOCH_SEC_SIZE * sampling_rate)]
    X = np.asarray(np.split(X_data_slim, n_epochs)).astype(np.float32)
    X = torch.from_numpy(X).float()
    # X ready
    return X

def data_x5(X):
    bs = X.shape[0]
    X5X = torch.ones([bs, 5, 3840, 6])
    X5X[0] = torch.cat((X[0:2], X[0:3]), 0)
    X5X[1] = torch.cat((X[0:1], X[0:4]), 0)
    X5X[bs - 1] = torch.cat((X[bs-3:bs], X[-2:]), 0)
    X5X[bs - 2] = torch.cat((X[bs-4:bs], X[-1:]), 0)
    for i in range(2, bs-2):
        X5X[i] = X[i-2:i+3]
    print(X5X.shape)
    # X5X shape: [length, 5, 3840, 6]
    X5X = X5X.view([-1, 19200, 6])
    X5X = torch.clip(X5X, -1000, 1000)
    X5X = torch.swapaxes(X5X, 1, 2)
    # X5X shape: [length, 6, 19200]
    return X5X

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
            step = n_epoch // 2   # 50% overlap
            batches = torch.zeros([L//(n_epoch//2), n_epoch, *C]).to(self.device)
            i_forward = -1
            while (i_forward + 2) * step < feat.shape[0]:
                i_forward += 1 
                batch = feat[i_forward * step:(i_forward + 2) * step]
                batches[i_forward][:batch.shape[0]] = batch
            return batches

    def batch_to_hypno(self, preds): 
        n_batch, n_epoch, *dims = preds.shape
        if n_batch == 1:
            pred_hyp = preds[0]
        else:
            step = n_epoch // 2   # 50% overlap
            pred_hyp = torch.zeros([(n_batch+1)*step, *dims]).to(self.device)
            for i in range(n_batch):
                pred_hyp[i*step:i*step+n_epoch] += torch.softmax(preds[i], 1)
        # pred_hyp = np.argmax(pred_hyp, axis = 1)
        return pred_hyp[:self.hypLen]

class ModelFeat(torch.nn.Module):
    def __init__(self, model_ep) -> None:
        super().__init__()
        self.feat_extr = model_ep.body
        self.downsample = model_ep.head.downsample
    def forward(self, x):
        x, _ = self.feat_extr(x)    # [bs, 240, 48]
        x = self.downsample(x)      # [bs, 30, 48]
        return x

def tester(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    model_ep = torch.load(cfg.best_ckp)['net'].cuda()
    model_hyp = torch.load(cfg.best_hyp_ckp)['net'].cuda()
    if type(model_ep) == DP:
        model_ep = model_ep.module
    if type(model_hyp) == DP:
        model_hyp = model_hyp.module
    model_ep.eval()
    model_hyp.eval()
    model_feat = ModelFeat(model_ep).cuda()
    model_feat.eval()

    X = data_generator(cfg.edf) # [len, chn]
    X5X = data_x5(X)
    Regulr = FeatRegularizer(cuda = True)
    

    with torch.no_grad():
        result_ep = []
        result_hyp = torch.empty([0, 1260, 6]).float().cuda()
        feat_deep = torch.empty([0, 30, 48]).float().cuda()

        # epoch scoring & deep feature
        for i in range(0, len(X5X), cfg.BATCH_SIZE):
            input = X5X[i: i + cfg.BATCH_SIZE]
            if len(input) == 0:
                break
            pred_ep, _ = model_ep(input.cuda())
            feat_deep = torch.cat([feat_deep, model_feat(input.cuda())])

            pred_ep = torch.argmax(pred_ep,1).cpu()
            result_ep += pred_ep.tolist()

        # merge feat
        feat_meta = psgMetaFeat(X.view(-1, 6)).float().cuda()
        feat_meta = feat_meta.unsqueeze(1).expand(
                            [feat_meta.shape[0], 30, feat_meta.shape[-1]]) 
        feature = torch.cat([feat_deep, feat_meta], 2)  #[n_ep, 30, 48+10*n_chn]

        # hyp scoring
        batches = Regulr.feat_to_batch(feature)
        for batch in batches:
            # forward TODO batch > 1
            pred, _ = model_hyp(batch.unsqueeze(0))
            result_hyp = torch.cat([result_hyp, pred])  
        result_hyp = Regulr.batch_to_hypno(result_hyp)
        result_hyp = torch.argmax(result_hyp, 1).cpu()
    return result_ep, result_hyp


if __name__ == "__main__":
    from tools.config_handler import yamlStruct
    # pars
    cfg = yamlStruct()
    cfg.add('CUDA_VISIBLE_DEVICES','0')
    cfg.add('experiment','default') 
    cfg.add('edf',r'\\192.168.31.100\Data\13_公开数据集\公开数据集\DOD\dod2edf\dod-h\21.edf')
    cfg.add('dataset','default')
    cfg.add('eval_parallel',False)
    cfg.add('tvt','all')
    cfg.add('eval_thread',2)
    cfg.add('PSG_EPOCH',1260)
    cfg.add('SWIN', yamlStruct())
    cfg.SWIN.add('EMBED_DIM',48)
    cfg.SWIN.add('IN_CHANS',6)
    cfg.SWIN.add('IN_LEN',19200)
    cfg.SWIN.add('OUT_CHANS',6)
    cfg.add('HEAD','stage150')
    cfg.add('BATCH_SIZE',180)
    cfg.add('freq',128)
    cfg.add('best_ckp',r"experiments\mixed_cohort\weights\best_checkpoint")
    cfg.add('best_hyp_ckp',r"experiments\mixed_cohort\weights_hyp\best_checkpoint")
    
    # inference
    results = tester(cfg)

    # plot
    import matplotlib.pyplot as plt
    plt.plot(results[0])
    plt.plot(results[1])
    plt.show()
    print('done')