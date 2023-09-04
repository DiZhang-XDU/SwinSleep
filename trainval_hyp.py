import os,time,pickle
import numpy as np
import torch
from timm import scheduler as schedulers    # do not delete!

from os.path import join
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import DataParallel as DP
from tools import *
from tools.dataset_feature import XY_dataset_feat
from sleep_models import build_model

import warnings
warnings.filterwarnings("ignore")##################################
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


def trainer(cfg):
    # environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    for p in [cfg.path.weights + '_hyp', cfg.path.tblogs + '_hyp']:
        if os.path.exists(p) is False:
            os.mkdir(p)
    seed = 0 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    criterion = criterions(cfg).cuda()

    # tensorboard
    import shutil
    if (not cfg.resume) and os.path.exists(cfg.path.tblogs + '_hyp'):
        shutil.rmtree(cfg.path.tblogs + '_hyp')
    writer = SummaryWriter(cfg.path.tblogs + '_hyp')
    
    # prepare dataloader 
    trainSet = XY_dataset_feat(cfg, tvt = 'train', train_trunc=True)
    validSet = XY_dataset_feat(cfg, tvt = 'valid', in_memory=True)

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = cfg.BATCH_SIZE, persistent_workers=True,
                                    shuffle = True, num_workers = cfg.train_thread, drop_last = False, pin_memory = True)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size = 1, persistent_workers=True,
                                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)

    # initialization
    model = build_model(cfg).cuda()
    if cfg.resume:
        optim = eval(cfg.optimizer)
        scheduler = eval(cfg.scheduler)
        loadObj = torch.load(join(cfg.path.weights, 'best_checkpoint'), map_location='cpu')
        model_, epoch_, optim_, scheduler_, best_loss_val_ = loadObj['net'], loadObj['epoch'], loadObj['optim'], loadObj['sched'], loadObj['best_loss_val']
        best_loss_val, best_f1_val, epoch = 9999, 0, 0
    else:
        optim = eval(cfg.optimizer)
        scheduler = eval(cfg.scheduler)
        best_loss_val, best_f1_val, epoch = 9999, 0, 0

    if cfg.train_parallel:
        model = DP(model) 

    print('start epoch')
    Regulr = FeatRegularizer(cuda = True)
    for epoch in range(epoch, cfg.EPOCH_MAX): 
        tic = time.time()
        name = ('train', 'valid')
        epoch_loss = {i:0 for i in name}
        epoch_acc = {i:0 for i in name}
        epoch_mf1 = {i:0 for i in name}

        record_target = {'train':torch.zeros(len(trainSet) * cfg.PSG_EPOCH) + 5, 
                        'valid':torch.LongTensor([])}
        record_pred = {'train':torch.zeros(len(trainSet) * cfg.PSG_EPOCH) + 5, 
                        'valid':torch.LongTensor([])}

        torch.cuda.empty_cache()
        model.train()
        tq = tqdm(trainLoader, desc= 'Trn', ncols=100, ascii=True)
        for idx, (feature, target, _) in enumerate(tq):
            # feature shape = [bs, 1260, 108]
            # forward
            feature = feature.cuda(non_blocking = True)
            pred, _ = model(feature)

            # loss
            pred = pred.reshape([-1, 6])
            target =  target.view(-1)
            loss = criterion(pred, target.cuda(non_blocking = True))
            if cfg.train_parallel:
                loss = torch.mean(loss)
            
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # record
            pred = torch.argmax(pred,-1).cpu()
            record_pred['train'][idx*cfg.PSG_EPOCH*cfg.BATCH_SIZE:
                                idx*cfg.PSG_EPOCH*cfg.BATCH_SIZE+pred.shape[0]] = pred
            record_target['train'][idx*cfg.PSG_EPOCH*cfg.BATCH_SIZE:
                                idx*cfg.PSG_EPOCH*cfg.BATCH_SIZE+pred.shape[0]] = target

            epoch_loss['train'] += loss.item()
            epoch_mf1['train'] += f1_score(target[target!=5], pred[target!=5], average='macro') 
            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['train'] / (tq.n+1)), 
                            'MF1:':'{:.4f}'.format(epoch_mf1['train'] / (tq.n+1))})
        
        epoch_mf1['train'] /= (idx+1)
        epoch_loss['train'] /= (idx+1)

        if (epoch + 1) % cfg.EPOCH_STEP != 0:
            continue
        # eval
        torch.cuda.empty_cache()
        model_eval = DP(model) if cfg.eval_parallel and not cfg.train_parallel else model
        model_eval.eval()
        
        with torch.no_grad():
            tq = tqdm(validLoader, desc = 'Val', ncols=100, ascii=True)
            for idx, (feature, target, _) in enumerate(tq):

                feature = torch.squeeze(feature, 0).cuda(non_blocking = True) # TODO batch > 1
                batches = Regulr.feat_to_batch(feature)

                pred_hyp = torch.empty([0, cfg.PSG_EPOCH, cfg.SWIN.IN_CHANS]).cuda()
                for batch in batches:
                    # forward TODO batch > 1
                    pred, _ = model(batch.unsqueeze(0))
                    pred_hyp = torch.cat([pred_hyp, pred])
                
                pred_hyp = Regulr.batch_to_hypno(pred_hyp)
                target = target.squeeze(0)

                # loss
                loss = criterion(pred_hyp, target.cuda(non_blocking = True))
                if cfg.train_parallel:
                    loss = torch.mean(loss)
                
                # record
                pred_hyp = torch.argmax(pred_hyp,-1).cpu()

                record_pred['valid'] = torch.cat([record_pred['valid'], pred_hyp])
                record_target['valid'] = torch.cat([record_target['valid'], target])

                epoch_loss['valid'] += loss.item()
                epoch_mf1['valid'] += f1_score(target[target!=5], pred_hyp[target!=5], average='macro')
                
                tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['valid'] / (idx+1)), 
                            'MF1:':'{:.4f}'.format(epoch_mf1['valid'] / (idx+1))})
                        
        epoch_loss['valid'] /= (idx+1)
        # epoch end now
        scheduler.step_update(epoch * cfg.EPOCH_STEP)

        # stat
        record_pred['train'] = record_pred['train'][record_pred['train'] != -1]
        record_target['train'] = record_target['train'][record_target['train'] != -1]
        assert len(record_pred['train']) == len(record_target['train'])
        assert -1 not in record_pred['valid']
        assert -1 not in record_target['valid']
        for idx in name:
            vidx = valued_pairs(record_target[idx], record_pred[idx])
            record_target[idx] = np.array(record_target[idx])[vidx]
            record_pred[idx] = np.array(record_pred[idx])[vidx]
            epoch_acc[idx] = accuracy_score(record_target[idx], record_pred[idx])
            epoch_mf1[idx] = f1_score(record_target[idx], record_pred[idx], average='macro')
        valid_kappa = cohen_kappa_score(record_target['valid'], record_pred['valid'])

        msg_epoch = 'epoch:{:02d}, time:{:2f}\n'.format(epoch, time.time() - tic)
        msg_loss = 'Trn Loss:{:.4f}, MF1:{:.2f}  Val Loss:{:.4f}, MF1:{:.2f}\n'.format(
            epoch_loss['train'], epoch_mf1['train'] * 100, 
            epoch_loss['valid'], epoch_mf1['valid'] * 100)
        
        msg_detail = classification_report(record_target['valid'], record_pred['valid']) \
                                + str(confusion_matrix(record_target['valid'], record_pred['valid'])) \
                                + '\nKappa:%.4f\nMF1:%.4f\nACC:%.4f\n\n'%(valid_kappa, epoch_mf1['valid'], epoch_acc['valid'])
        print(msg_epoch + msg_loss[:-1] + msg_detail)

        # save
        writer.add_scalars('Loss',{'train':epoch_loss['train'] , 'valid':epoch_loss['valid']},epoch)
        writer.add_scalars('Acc',{'train':epoch_acc['train'], 'valid':epoch_acc['valid']},epoch)
        writer.add_scalars('MF1',{'train':epoch_mf1['train'], 'valid':epoch_mf1['valid']},epoch)

        with open(cfg.path.log.replace('log.txt','log_hyp.txt'), 'a') as f:
            f.write(msg_epoch)
            f.write(msg_loss)
            f.write(msg_detail)
        if best_f1_val < epoch_mf1['valid']:
            best_f1_val = epoch_mf1['valid']
            saveObj = {'net': model, 'epoch':epoch, 'optim':optim , 'sched':scheduler, 'best_loss_val':best_loss_val}
            torch.save(saveObj, join(cfg.path.weights + '_hyp', 'best_checkpoint'))
        torch.save(saveObj, join(cfg.path.weights + '_hyp', 'epoch_{:02d}_val_loss={:4f}_mf1={:.4f}'.format(epoch, epoch_loss['valid'], epoch_mf1['valid'])))
            
    writer.close()

if __name__ == "__main__":
    trainer()