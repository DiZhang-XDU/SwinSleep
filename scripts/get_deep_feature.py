import os, sys, pickle, mne
sys.path.append(os.getcwd())
from scipy import stats
from os.path import join
from tqdm import tqdm
from tools import *
from sleep_models.swin_heads import GAP_head
from torch.nn import DataParallel as DP
import warnings
warnings.filterwarnings("ignore")

ch_types = ['C4','F4','O2','E1','E2','EMG']

class ModelFeat(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model_ep = torch.load(cfg.best_ckp)['net']
        if type(model_ep) == DP:
            model_ep = model_ep.module
        self.feat_extr = model_ep.body
        self.downsample = model_ep.head.downsample
    def forward(self, x):
        x, _ = self.feat_extr(x)    # [bs, 240, 48]
        x = self.downsample(x)      # [bs, 30, 48]
        return x

def getDeepFeat(cfg):
    deepFeatDir = join('experiments/%s/prepared_data/train_deep_feature'%(cfg.experiment))

    # init model
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    model = ModelFeat(cfg).cuda()
    if cfg.eval_parallel:
        model = DP(model)
    model.eval()

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
        saveDir = deepFeatDir.replace('train',dataset.tvt)
        if os.path.exists(saveDir):
            print('%s deep features exist.'%dataset.tvt)
            continue
        os.mkdir(saveDir)

        dataLoader = torch.utils.data.DataLoader(dataset, batch_size = 1, 
                                    shuffle = False, num_workers = 6, 
                                    drop_last = False, pin_memory = True)
        
        tq = tqdm(dataLoader, desc= 'loading.', ncols=80, ascii=True)
        for i, (psg_signal, target, subjs) in enumerate(tq):
            psg_signal = torch.squeeze(psg_signal, 0)   # TODO batch > 1
            L, C = psg_signal.shape
            L = L // (cfg.freq * 30) 
            
            # psg X5
            psg5x =  torch.zeros([L, cfg.SWIN.IN_LEN, cfg.SWIN.OUT_CHANS]).float()
            padding = torch.zeros([2 * 30 * cfg.freq, C])
            psg_ex = torch.cat([padding,
                                psg_signal,
                                padding],dim=0)
            for i in range(0, psg5x.shape[0]):
                psg5x[i] = psg_ex[i*30*cfg.freq : i*30*cfg.freq + cfg.SWIN.IN_LEN]

            # inference
            with torch.no_grad():
                feature_psg = torch.zeros([0, 30, cfg.SWIN.EMBED_DIM]).cuda() + torch.inf
                psg5x = psg5x.swapaxes(1,2).cuda()    # psg5x shape = [L, C, 19200]
                step = cfg.PSG_EPOCH // 2
                for i in range(0, L, step):
                    p = model(psg5x[i:i+step])
                    feature_psg = torch.cat([feature_psg, p])
                assert feature_psg.shape[0] == L and torch.inf not in feature_psg
                feature_psg = feature_psg.cpu()
            torch.cuda.empty_cache()

            # save
            with open(join(saveDir, '%04d.pkl'%(int(subjs))), 'wb') as f:
                pickle.dump(feature_psg, f)     #[n_ep, 30, 48]
        
        print(dataset.tvt, 'done!')


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
    cfg.add('eval_parallel', False)
    cfg.add('redir_cache', None)
    cfg.add('redir_root',["G:/data/filtered_data_128/subjects", "E:/data/filtered_data_128/subjects"])
    cfg.add('best_ckp',r"experiments/mixed_cohort/weights/best_checkpoint")

    getDeepFeat(cfg)
