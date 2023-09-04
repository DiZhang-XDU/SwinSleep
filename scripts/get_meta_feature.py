import os, sys, pickle, threading, time
sys.path.append(os.getcwd())
from scipy import stats, signal
from os.path import join
from tqdm import tqdm
from tools import *
import warnings
warnings.filterwarnings("ignore")

_default_stdout = sys.stdout
ch_types = ['C4','F4','O2','E1','E2','EMG']
fbands = {'low':[0, .5],
        'delta':[.5, 4],
        'theta':[4, 8],
        'alpha':[8, 15],
        'sigma':[12, 16],
        'beta':[16, 31]}    # hfreq = 35
f_ = np.linspace(0,64,129)
fmasks = [np.logical_and(f_>=low, f_<=high) for (_, (low, high)) in fbands.items()]

def psgMetaFeat(X:torch.Tensor): # X shape: [len, chn]
    
    X = X.swapaxes(0,1).numpy()
    n_ep = int(X.shape[1] // (30*128))
    len_ep = 128 * 30

    # formward
    feature_spec = np.zeros([n_ep, 6, len(ch_types)]) + np.inf
    feature_temp = np.zeros([n_ep, 4, len(ch_types)]) + np.inf
    

    for ep in range(n_ep):
        x = X[:, ep*len_ep:(ep+1)*len_ep]

        # spec feat
        _, spec = signal.welch(x, 128, nperseg=256)
        for (i, (_, value)) in enumerate(fbands.items()):
            sband = spec[:,fmasks[i]]
            feature_spec[ep][i][:] = np.mean(sband, 1)

        # temp feat
        feature_temp[ep][0][:] = (np.max(x, 1) - np.min(x, 1)) / 30     # amplitude
        feature_temp[ep][1][:] = np.std(x, 1)                           # Standard Deviation
        feature_temp[ep][2][:] = stats.skew(x, 1)                       # Skewness
        feature_temp[ep][3][:] = stats.kurtosis(x, 1)                   # Kurtosis
    # feature_spec = 10 * np.log10(np.maximum(feature_spec, np.finfo(float).tiny))
    
    # merge feat
    assert np.inf not in feature_spec
    feature_meta = np.concatenate((feature_spec, feature_temp), axis = 1)
    feature_meta = feature_meta.reshape((feature_meta.shape[0], -1))
    feature_meta = torch.from_numpy(feature_meta)   # [n_ep, 10 * n_chn]
    
    # output
    feature_meta = torch.nan_to_num(feature_meta)
    torch.clip_(feature_meta, -1000, 1000)
    return feature_meta

def getMetaFeat(cfg):
    metaFeatDir = join('experiments/%s/prepared_data/train_meta_feature.pkl'%(cfg.experiment))
    
    # init data
    myDataset = datasets(cfg)

    # process
    if cfg.tvt == 'all':
        datasets_ = [myDataset(cfg, tvt='all')]
    else:
        datasets_ = [myDataset(cfg, tvt='train', train_trunc=False),
                    myDataset(cfg, tvt='valid'),
                    myDataset(cfg, tvt='test')]
    for dataset in datasets_:
        saveDir = metaFeatDir.replace('train', dataset.tvt)
        if os.path.exists(saveDir):
            continue

        dataLoader = torch.utils.data.DataLoader(dataset, batch_size = 1, 
                                    shuffle = False, num_workers = 1, 
                                    drop_last = False, pin_memory = True)
        
        feature_meta = {}
        # tq = tqdm(dataLoader, desc= 'loading.', ncols=80, ascii=True)
        for i, (psg_signal, target, psg_idx) in enumerate(dataLoader):
            psg_signal = torch.squeeze(psg_signal, 0)   # TODO batch > 1

            # meta feat
            feature_psg = psgMetaFeat(psg_signal)
            feature_meta[int(psg_idx)] = feature_psg
            print('%d/%d: %s done!  %s'%(i, len(dataset), int(psg_idx), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            print(feature_psg.shape)

        # save
        with open(saveDir, 'wb') as f:
            pickle.dump(feature_meta, f)
        

if __name__ == "__main__":
    from tools.config_handler import yamlStruct
    # pars
    cfg = yamlStruct()
    cfg.add('tvt','null')
    cfg.add('dataset',["SHHS1","SHHS2","CCSHS","SOF","CFS","MROS1","MROS2","MESA","HPAP1","HPAP2","ABC","NCHSDB","MASS13","HMC","SSC","CNC","PHY","WSC"])
    cfg.add('CUDA_VISIBLE_DEVICES','0')
    cfg.add('SWIN', yamlStruct())
    cfg.SWIN.add('EMBED_DIM',48)
    cfg.SWIN.add('IN_CHANS',6)
    cfg.SWIN.add('OUT_CHANS',6)
    cfg.SWIN.add('IN_LEN', 19200)
    cfg.SWIN.add('WINDOW_SIZE', 8)
    cfg.add('experiment','mixed_cohort')
    cfg.add('HEAD','stagepsg')
    cfg.add('PSG_EPOCH',1260)
    cfg.add('freq',128)
    cfg.add('eval_parallel', True)
    cfg.add('eval_thread', 2)
    cfg.add('redir_cache', None)
    cfg.add('redir_root',["G:/data/filtered_data_128/subjects", "E:/data/filtered_data_128/subjects"])
    cfg.add('best_ckp',r"experiments/mixed_cohort/weights/best_checkpoint")

    getMetaFeat(cfg)
