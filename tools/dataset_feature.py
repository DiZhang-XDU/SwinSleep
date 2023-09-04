import torch
import torch.utils.data as data
import os
import pickle
from os.path import join
from tools import FeatAugment

class XY_dataset_feat(data.Dataset):
    def __init__(self, cfg, tvt = 'train', train_trunc = True, in_memory = False):
    ###
    # tvt: 'train', 'valid', 'test', 'all'
    ###
        super(XY_dataset_feat, self).__init__()
        self.tvt = tvt
        self.PSG_EPOCH = cfg.PSG_EPOCH
        self.train_trunc = train_trunc
        self.in_memory = in_memory
        self.augment = FeatAugment()
        # torch.manual_seed(0)

        # cache dir
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
        
        # feat dir
        feat_deep_dir ='experiments/{:}/prepared_data/{:}_deep_feature'.format(cfg.experiment, tvt)
        feat_meta_path = 'experiments/{:}/prepared_data/{:}_meta_feature.pkl'.format(cfg.experiment, tvt)
        assert len(os.listdir(feat_deep_dir))>0 and os.path.exists(feat_meta_path)
        with open(feat_meta_path, 'rb') as f:
            feature_meta = pickle.load(f)
        
        # prepare mixed feature
        self.feature_meta = []
        self.feature_deep = []
        self.y = []
        if os.path.exists(cache_path):          
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if not redir_root:
                self.root = cache['root']
            else:
                self.root = redir_root

            self.index_list = list(set(cache['items_subj']))
            for idx in self.index_list:
                # load y
                r = self.root[0] if idx < 15000 else self.root[1]
                if os.path.exists('{:}\\{:06d}\\stages.pkl'.format(r, idx)):
                    with open('{:}\\{:06d}\\stages.pkl'.format(r, idx), 'rb') as f:
                        y = pickle.load(f)
                else:
                    y = None
                self.y.append(y)

                # load deep feature
                f_path = join(feat_deep_dir, '%04d.pkl'%(idx))  
                # f_path = f_path.replace('experiments/','G:/') if idx%2 == 1 else f_path ############
                if in_memory:
                    with open(f_path, 'rb') as f:
                        feature_deep = pickle.load(f)           #[n_ep, 30, 48]
                    self.feature_deep.append(feature_deep)
                else:
                    # deep feature will be loaded in __getitem__()
                    self.feature_deep.append(f_path)

                # load meta feature
                self.feature_meta.append(feature_meta[idx])
        else:
            assert 0 #TODO   TODO    TODO
        self.len = len(self.index_list)

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
        # idx = self.index_list[index]
        idx = index
        # load data
        if not self.in_memory:
            with open(self.feature_deep[idx], 'rb') as f:
                feature_deep = pickle.load(f)       #[n_ep, 30, 48]
        else:
            feature_deep = self.feature_deep[idx]   #[n_ep, 30, 48]
        feature_deep = torch.tanh(feature_deep )  #####################

        feature_meta = self.feature_meta[idx]       #[n_ep, 10*n_chn]
        # if self.tvt == 'train':                     # meta feature augmentation
        #     feature_meta = self.augment(feature_meta)   

        y = self.y[idx] if self.y[idx] is not None \
            else torch.zeros(feature_meta.shape[0]).long() + 5  #[n_ep] (0~5)

        # merge feature
        feature_meta = feature_meta.unsqueeze(1).expand(
                    [feature_meta.shape[0],30,feature_meta.shape[-1]]) 
        X = torch.cat([feature_deep, feature_meta], 2)  #[n_ep, 30, 48+10*n_chn]
        
        # trunc
        if self.tvt == 'train' and self.train_trunc:
            start = self._trunc(self.y[index])
            end = min(X.shape[0], start + self.PSG_EPOCH)
            X = torch.cat([X[start:end], torch.zeros([self.PSG_EPOCH-(end-start),*X.shape[1:]])])
            y = torch.cat([y[start:end], torch.zeros([self.PSG_EPOCH-(end-start)])+5])     # unknown = 5
        else:
            pass        # TODO: bs>1 in eval

        # clipped meta_feat and NOT clipped deep_feat
        # X = torch.clip(X, -1000, 1000)    
        return X.float(), y.long(), int(self.index_list[index])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass
