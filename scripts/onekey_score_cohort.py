import warnings
import numpy as np
import torch,os
from os.path import join
from torch.autograd import Variable
from torch.nn import DataParallel as DP
from mne.io import read_raw_edf

# set env
np.set_printoptions(suppress=True) 
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
multiprocessing.set_start_method('spawn', True)
# set path
import os, sys
sys.path.append(os.getcwd())

from tools import *
from test_stage import tester_5ep
from test_hyp import tester_hyp

if __name__ == "__main__":
    from tools.config_handler import yamlStruct
    # pars
    cfg = yamlStruct()
    cfg.add('CUDA_VISIBLE_DEVICES','0')
    # cfg.add('experiment','mixed_cohort')
    cfg.add('experiment','eval_daytime')
    # cfg.add('experiment','eval_consensus')
    # cfg.add('dataset',["SHHS1","SHHS2","CCSHS","SOF","CFS","MROS1","MROS2","MESA","HPAP1","HPAP2","ABC","NCHSDB","MASS13","HMC","SSC","CNC","PHY","WSC"])
    cfg.add('dataset',['DCSM','DHC'])
    # cfg.add('dataset',['DOD','ISRC'])
    cfg.add('eval_parallel',False)
    cfg.add('tvt','all')
    cfg.add('redir_cache',None)
    cfg.add('redir_root',["G:/data/filtered_data_128/subjects", "E:/data/filtered_data_128/subjects"])
    cfg.add('eval_thread',2)
    cfg.add('PSG_EPOCH',1260)
    cfg.add('SWIN', yamlStruct())
    cfg.SWIN.add('EMBED_DIM',48)
    cfg.SWIN.add('IN_CHANS',6)
    cfg.SWIN.add('IN_LEN',19200)
    cfg.SWIN.add('OUT_CHANS',6)
    cfg.add('HEAD','stage150')
    cfg.add('BATCH_SIZE',128)
    cfg.add('freq',128)
    cfg.add('best_ckp',r"experiments\mixed_cohort\weights\best_checkpoint")
    cfg.add('best_hyp_ckp',r"experiments\mixed_cohort\weights_hyp\best_checkpoint")

    from get_meta_feature import getMetaFeat
    from get_deep_feature import getDeepFeat

    # start ep scoring
    result = tester_5ep(cfg)
    
    # start hyp scoring
    cfg.HEAD = 'stagepsg'
    getDeepFeat(cfg)
    getMetaFeat(cfg)
    cfg.eval_thread = 0
    tester_hyp(cfg)
    
    print('done')