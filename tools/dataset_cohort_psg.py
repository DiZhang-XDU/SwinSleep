import torch
import torch.utils.data as data
import os
import pickle
from os.path import join
from sklearn.model_selection import train_test_split

class XY_dataset_N2One(data.Dataset):
    def __init__(self, cfg, tvt = 'train', serial_len = 1260, train_trunc = True):
    ###
    # tvt: 'train', 'valid', 'test', 'all'
    ###
        super(XY_dataset_N2One, self).__init__()
        self.serial_len = serial_len
        self.tvt = tvt
        self.train_trunc = train_trunc
        self.frame_len = cfg.freq * 30
        self.channel_num = cfg.SWIN.IN_CHANS
        dataset = cfg.dataset
        if type(dataset) is str:
            savName = dataset
            dataset = [dataset]
        else:
            savName = 'Custom{:02d}'.format(len(dataset))
        redir_cache, redir_root = cfg.redir_cache, cfg.redir_root
        if not redir_cache:
            cache_path = join('experiments/{:}/prepared_data/{:}_{:}_cache.pkl'.format(cfg.experiment, tvt, savName))
        else:
            cache_path = join(redir_cache, '{:}_{:}_cache.pkl'.format(tvt, savName))
        
        # cache
        if os.path.exists(cache_path):          
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if not redir_root:
                self.root = cache['root']
            else:
                self.root = redir_root

            psg_list = list(set(cache['items_subj']))
            self.psg_path, self.psg_len, self.y = [], [], []
            for idx in psg_list:
                r = self.root[0] if idx < 15000 else self.root[1]
                psg_path = '{:}\\{:06d}\\data\\'.format(r, idx)
                self.psg_path.append(psg_path)
                self.psg_len.append(len(os.listdir(psg_path)))
                if os.path.exists('{:}\\{:06d}\\stages.pkl'.format(r, idx)):
                    with open('{:}\\{:06d}\\stages.pkl'.format(r, idx), 'rb') as f:
                        y = pickle.load(f)
                else:
                    y = torch.zeros(len(os.listdir(psg_path))).long()
                self.y.append(y)
        else:
            assert 0 #TODO   TODO    TODO
        self.len = len(psg_list)

    def _trunc(self, hyp, clip_len = 1260):
    # return start index
        if len(hyp) > clip_len:
            sleep = torch.where(abs(hyp-2.5)<2)[0]
            # 1
            if len(sleep) <= clip_len:
                offset_from = max(0, len(sleep) - clip_len)
                offset_to = sleep[0] + 1
                return torch.randint(offset_from, offset_to, [1])
            # 2
            else:
                return torch.randint(len(hyp) - clip_len, [1])
        return 0

    def __getitem__(self, index):
        psg_path = self.psg_path[index]
        psg_len = self.psg_len[index]
        if self.tvt == 'train' and self.train_trunc:
            X = torch.zeros(size = [self.serial_len * self.frame_len, self.channel_num]).float()
            y = torch.zeros(size = [self.serial_len]).long() - 1
            start = self._trunc(self.y[index])
            end = min(psg_len, self.serial_len)
        else:
            X = torch.zeros(size = [psg_len * self.frame_len, self.channel_num]).float()
            y = torch.zeros(size = [psg_len]).long()
            start = 0
            end = psg_len

        for i in range(end):
            with open('%s%04d.pkl'%(psg_path, i + start), 'rb') as f_data:
                pkl = pickle.load(f_data)
            # train: pad0/trunc to end
            # eval: no shift
            X[i*self.frame_len:(i+1)*self.frame_len,:] = torch.from_numpy(pkl).float()
            y[i] = self.y[index][i]

        X = torch.clip(X, -1000, 1000)

        return X, y, int(psg_path[-12:-6])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass