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
from sleep_models import build_model

import warnings
warnings.filterwarnings("ignore")##################################
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class validDatasetInMemory(torch.utils.data.Dataset):
    def __init__(self, validSet:torch.utils.data.Dataset) -> None:
        super().__init__()
        self.validSet = validSet
        self.X = torch.zeros(size = [len(validSet), validSet.frame_len, validSet.channel_num]).float()
        self.y = torch.zeros(size = [len(validSet)]).long()
        # load data
        self.psgStartList = []
        self.subjList = []
        lastSubj = -1
        for index in range(len(validSet)):
            subj = validSet.items_psg[index]
            idx = validSet.items_idx[index]
            root = validSet.root[0] if subj<15000 else validSet.root[1] 
            path = '{:}\\{:06d}\\data\\{:04d}.pkl'.format(root, subj, idx)
            with open(path, 'rb') as f_data:
                pkl = pickle.load(f_data)
            self.X[index] = torch.from_numpy(pkl).float()
            self.y[index] = torch.tensor(validSet.y[index]).long()
            if lastSubj != subj:
                lastSubj = subj
                self.psgStartList += [index]
                self.subjList += [subj]
        self.psgStartList += [index + 1]

    def __getitem__(self, index):
        if index in self.psgStartList:
            seq_idx = [index, index, index, index + 1, index + 2]
        elif index - 1 in self.psgStartList:
            seq_idx = [index - 1, index - 1, index, index + 1, index + 2]
        elif index + 1 in self.psgStartList:
            seq_idx = [index - 2, index - 1, index, index, index]
        elif index + 2 in self.psgStartList:
            seq_idx = [index - 2, index - 1, index, index + 1, index + 1]
        else:
            seq_idx = [index - 2, index - 1, index, index + 1, index + 2]
        X = self.X[seq_idx].view([-1, self.validSet.channel_num])
        X = torch.clip(X, -1000, 1000)
        X = torch.swapaxes(X, 0, 1)
        y = self.y[index]
        subj = self.validSet.items_psg[index]
        return X, y, subj

    def __len__(self):
        return len(self.y)

def trainer(cfg):
    # environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    for p in [cfg.path.weights, cfg.path.tblogs]:
        if os.path.exists(p) is False:
            os.mkdir(p)
    seed = 0 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    criterion = criterions(cfg).cuda()

    # tensorboard
    import shutil
    if (not cfg.resume) and os.path.exists(cfg.path.tblogs):
        shutil.rmtree(cfg.path.tblogs)
    writer = SummaryWriter(cfg.path.tblogs)
    
    # prepare dataloader 
    myDataset = datasets(cfg)
    trainSet = myDataset(cfg, tvt = 'train')
    validSet = myDataset(cfg, tvt = 'valid')
    # validSet = validDatasetInMemory(validSet)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = cfg.BATCH_SIZE, persistent_workers=True,
                                    shuffle = True, num_workers = cfg.train_thread, drop_last = False, pin_memory = True)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size = cfg.BATCH_SIZE, persistent_workers=True,
                                    shuffle = False, num_workers = cfg.eval_thread, drop_last = False, pin_memory = True)

    # model initialization
    model = build_model(cfg).cuda()
    if cfg.resume:
        optim = torch.optim.SGD(model.parameters(), lr= 1e-1, momentum = .9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,4,1e-3)
        loadObj = torch.load(join(cfg.path.weights, 'checkpoint_'), map_location='cpu')
        model_, epoch_, optim_, scheduler_, best_loss_val_ = loadObj['net'], loadObj['epoch'], loadObj['optim'], loadObj['sched'], loadObj['best_loss_val']
        model.load_state_dict(model_.state_dict())
        optim.load_state_dict(optim_.state_dict())
        best_loss_val, best_f1_val, epoch = 9999, 0, 0
    else:
        optim = eval(cfg.optimizer)
        scheduler = eval(cfg.scheduler)
        best_loss_val, best_f1_val, epoch = 9999, 0, 0
    if cfg.train_parallel:
        model = DP(model)
    augment = DataAugment().cuda()
    augment.eval()

    print('start epoch')
    step = 0
    trainIter = iter(trainLoader)   
    for epoch in range(epoch, cfg.EPOCH_MAX): 
        tic = time.time()
        name = ('train', 'valid')
        epoch_loss = {i:0 for i in name}
        epoch_acc = {i:0 for i in name}
        epoch_mf1 = {i:0 for i in name}

        record_target = {'train':torch.zeros(cfg.EPOCH_STEP * cfg.BATCH_SIZE) - 1, 
                        'valid':torch.zeros(len(validSet)) - 1}
        record_pred = {'train':torch.zeros(cfg.EPOCH_STEP * cfg.BATCH_SIZE) - 1, 
                        'valid':torch.zeros(len(validSet)) - 1}

        torch.cuda.empty_cache()
        model.train()
        tq = tqdm(range(cfg.EPOCH_STEP), desc= 'Trn', ncols=80, ascii=True)
        # with torch.autograd.profiler.profile(enabled=True) as prof:
        for i, _ in enumerate(tq):
            input, target, _ = next(trainIter)
            step += 1
            if step == len(trainLoader):
                step = 0
                trainIter = iter(trainLoader)

            # data augment in training
            input = input.cuda(non_blocking = True)
            with torch.no_grad():
                input = augment(input)
            input.requires_grad = True

            # forward
            pred, _ = model(input)
            loss = criterion(pred, target.cuda(non_blocking = True))
            if cfg.train_parallel:
                loss = torch.mean(loss)
            
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # record
            pred = torch.argmax(pred,1).cpu()
            record_pred['train'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+pred.shape[0]] = pred
            record_target['train'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+pred.shape[0]] = target

            epoch_loss['train'] += loss.item()
            epoch_mf1['train'] += f1_score(target[target!=5], pred[target!=5], average='macro')
            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['train'] / (tq.n+1)), 
                            'MF1:':'{:.4f}'.format(epoch_mf1['train'] / (tq.n+1))})
        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        epoch_mf1['train'] /= (i+1)
        epoch_loss['train'] /= (i+1)

        # eval
        torch.cuda.empty_cache()
        model_eval = DP(model) if cfg.eval_parallel and not cfg.train_parallel else model
        model_eval.eval()
        
        with torch.no_grad():
            tq = tqdm(validLoader, desc = 'Val', ncols=75, ascii=True)
            for i, (input, target, _) in enumerate(tq):
                with torch.no_grad(): 
                    pred, _ = model_eval(input.cuda(non_blocking = True))
                loss = criterion(pred, target.cuda(non_blocking = True))

                if cfg.eval_parallel:
                    loss = torch.mean(loss)
                
                # record
                pred = torch.argmax(pred,1).cpu()
                record_pred['valid'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+cfg.BATCH_SIZE] = pred
                record_target['valid'][i*cfg.BATCH_SIZE:i*cfg.BATCH_SIZE+cfg.BATCH_SIZE] = target


                epoch_loss['valid'] += loss.item()
                epoch_mf1['valid'] += f1_score(target[target!=5], pred[target!=5], average='macro')
                
                
                tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['valid'] / (i+1)), 
                            'MF1:':'{:.4f}'.format(epoch_mf1['valid'] / (i+1))})
                        
        epoch_loss['valid'] /= (i+1)
        
        # epoch end
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
        msg_loss = 'Trn Loss:{:.4f}, acc:{:.2f}  Val Loss:{:.4f}, acc:{:.2f}\n'.format(
            epoch_loss['train'], epoch_acc['train'] * 100, 
            epoch_loss['valid'], epoch_acc['valid'] * 100)
        
        msg_detail = classification_report(record_target['valid'], record_pred['valid']) \
                                + str(confusion_matrix(record_target['valid'], record_pred['valid'])) \
                                + '\nKappa:%.4f\nMF1:%.4f\nACC:%.4f\n\n'%(valid_kappa, epoch_mf1['valid'], epoch_acc['valid'])
        print(msg_epoch + msg_loss[:-1] + msg_detail)

        # save
        writer.add_scalars('Loss',{'train':epoch_loss['train'] , 'valid':epoch_loss['valid']},epoch)
        writer.add_scalars('Acc',{'train':epoch_acc['train'], 'valid':epoch_acc['valid']},epoch)
        writer.add_scalars('MF1',{'train':epoch_mf1['train'], 'valid':epoch_mf1['valid']},epoch)

        with open(cfg.path.log, 'a') as f:
            f.write(msg_epoch)
            f.write(msg_loss)
            f.write(msg_detail)
        if best_f1_val < epoch_mf1['valid']:
            best_f1_val = epoch_mf1['valid']
            saveObj = {'net': model, 'epoch':epoch, 'optim':optim , 'sched':scheduler, 'best_loss_val':best_loss_val}
            torch.save(saveObj, join(cfg.path.weights, 'best_checkpoint'))
        torch.save(saveObj, join(cfg.path.weights, 'epoch_{:02d}_val_loss={:4f}_mf1={:.4f}'.format(epoch, epoch_loss['valid'], epoch_acc['valid'])))
            
    writer.close()

if __name__ == "__main__":
    pass