import torch
import torch.utils.data as data
import os
import pickle
from os.path import join
from sklearn.model_selection import train_test_split

class DatasetInMemory(torch.utils.data.Dataset):
    def __init__(self, validSet:torch.utils.data.dataset.Subset) -> None:
        super().__init__()
        self.validSet = validSet
        self.X = torch.zeros(size = [len(validSet.indices), validSet.dataset.frame_len, validSet.dataset.channel_num]).float()
        self.y = torch.zeros(size = [len(validSet.indices)]).long()
        # load data
        self.psgStartList = []
        self.subjList = []
        lastSubj = -1
        for index in range(len(validSet.indices)):
            subj = validSet.dataset.items_subj[validSet.indices[index]]
            idx = validSet.dataset.items_idx[validSet.indices[index]]
            root = validSet.dataset.root[0] if subj<15000 else validSet.dataset.root[1] 
            path = '{:}\\{:06d}\\data\\{:04d}.pkl'.format(root, subj, idx)
            with open(path, 'rb') as f_data:
                pkl = pickle.load(f_data)
            self.X[index] = torch.from_numpy(pkl).float()
            self.y[index] = torch.tensor(validSet.dataset.y[validSet.indices[index]]).long()
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
        # seq_idx = [index, index, index, index, index]
        X = self.X[seq_idx].view([-1, self.validSet.dataset.channel_num])
        X = torch.clip(X, -1000, 1000)
        X = torch.swapaxes(X, 0, 1)
        y = self.y[index]
        subj = self.validSet.dataset.items_subj[self.validSet.indices[index]]
        return X, y, subj

    def __len__(self):
        return len(self.validSet.indices)

class XY_dataset_N2One(data.Dataset):
    def __init__(self, cfg, tvt = 'train', serial_len = 5):
    ###
    # tvt: 'train', 'valid', 'test', 'all'
    ###
        super(XY_dataset_N2One, self).__init__()
        self.serial_len = serial_len
        self.tvt = tvt
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
        #if cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if not redir_root:
                self.root = cache['root']
            else:
                self.root = redir_root
            self.items_psg = cache['items_subj']
            self.items_idx = cache['items_idx']
            self.boundary = cache['fences']
            self.y = cache['y']
            self.len = len(self.items_psg)
            return
        #else
        
        # subject selector
        if not redir_root:
            self.root = [r'G:\data\filtered_data_128\subjects', r'E:\data\filtered_data_128\subjects']
        else:
            self.root = redir_root
        self.subjIdx = {
                'SHHS1': (0, 5666),
                'SHHS2': (5667, 8287),
                'CCSHS': (8288, 8802),
                'SOF': (8803, 9251),
                'CFS': (9252, 9973),
                'MROS1': (9974, 12851),
                'MROS2': (12852, 13859),
                'MESA': (13860, 15893),
                'HPAP1':(15894, 16083),
                'HPAP2':(16084, 16138),
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
                'SHHS1ex':(19777,19902),
                'WSC':(19903, 22436),
                'ISRC':(22437, 22505)
                }

        # split
        psg_paths = []
        if tvt == 'all':
            for d in dataset:
                for i in range(self.subjIdx[d][0], self.subjIdx[d][1] + 1):
                    root = self.root[0] if i<15000 else self.root[1]
                    psg_path = join(root, '{:06d}'.format(i))
                    assert os.path.exists(psg_path)
                    psg_paths.append(psg_path)
        else:
            from tools.data_tools import Split
            train_idx, valid_idx, test_idx = Split().split_dataset(dataset)
            psg_idx = train_idx if tvt == 'train' else valid_idx if tvt == 'valid' else test_idx if tvt == 'test' else None
            for idx in psg_idx:
                root = self.root[0] if idx<15000 else self.root[1]
                psg_path = join(root, '{:06d}'.format(idx))
                assert os.path.exists(psg_path)
                psg_paths.append(psg_path)

        # generate idx
        self.items_psg, self.items_idx, self.boundary, self.y = [], [], [0], []
        for psg_path in psg_paths:
            frameNum = len(os.listdir(join(psg_path, 'data')))
            if os.path.exists(join(psg_path, 'stages.pkl')):
                with open(join(psg_path, 'stages.pkl'), 'rb') as f:
                    anno = pickle.load(f)
            else:
                anno = torch.zeros(frameNum)
            for i in range(frameNum):
                self.items_idx.append(i)
                self.items_psg.append(int(psg_path[-6:]))
                self.y.append(int(anno[i]))
            self.boundary += [len(self.y)]

        self.len = len(self.items_psg)
        # save cache. TODO: replace variable name: 'items_subj', 'fences'
        cache = {'root':self.root, 'items_subj': self.items_psg,'items_idx':self.items_idx, 'fences':self.boundary, 'y': self.y}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def _random_psg_split(self, subj_paths):
        train_paths, valid_paths = train_test_split(subj_paths, train_size = 0.8, random_state = 0)
        valid_paths, test_paths = train_test_split(valid_paths, train_size = 0.5, random_state = 0)
        
        tvt2paths = {'train':train_paths ,'valid':valid_paths, 'test':test_paths, 'all':subj_paths}
        return tvt2paths[self.tvt]

    def __getitem__(self, index):
        # with torch.autograd.profiler.profile(enabled=True) as prof:
        assert self.serial_len == 5 #TODO
        index_pkl = self.items_idx[index]
        # TODO run too slow in this step 
        if index in self.boundary:
            seq_idx = [index_pkl, index_pkl, index_pkl, index_pkl + 1, index_pkl + 2]
        elif index - 1 in self.boundary:
            seq_idx = [index_pkl - 1, index_pkl - 1, index_pkl, index_pkl + 1, index_pkl + 2]
        elif index + 1 in self.boundary:
            seq_idx = [index_pkl - 2, index_pkl - 1, index_pkl, index_pkl, index_pkl]
        elif index + 2 in self.boundary:
            seq_idx = [index_pkl - 2, index_pkl - 1, index_pkl, index_pkl + 1, index_pkl + 1]
        else: 
            seq_idx = [index_pkl - 2, index_pkl - 1, index_pkl, index_pkl + 1, index_pkl + 2]
        subj = self.items_psg[index]
        root = self.root[0] if subj<15000 else self.root[1] 
        paths = ['{:}\\{:06d}\\data\\{:04d}.pkl'.format(root, subj, idx) for idx in seq_idx]
        X = torch.zeros(size = [self.serial_len * self.frame_len, self.channel_num]).float()
        for i in range(self.serial_len):
            with open(paths[i], 'rb') as f_data:
                pkl = pickle.load(f_data)
            X[i*self.frame_len:(i+1)*self.frame_len,:] = torch.from_numpy(pkl).float()
        X = torch.clip(X, -1000, 1000)
        X = torch.swapaxes(X, 0, 1)
        y = torch.tensor(self.y[index]).long()

        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        return X, y, subj

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass