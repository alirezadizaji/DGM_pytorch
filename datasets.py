from typing import Callable, List, Optional
import torch
import pickle
import numpy as np
import os.path as osp
import torch
import pandas as pd

from enums.separation import DataSeparation
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class UKBBAgeDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu'):
        with open('data/UKBB.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data

        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)
        if train:
            self.mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        else:
            self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask
    
    
    
class TadpoleDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu',full=False):
        with open('data/tadpole_data.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        
        if not full:
            X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.

        
        self.n_features = X_.shape[-2]
        self.num_classes = y_.shape[-2]
        
        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)
        if train:
            self.mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        else:
            self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask, [[]]
 
    
# class TadpoleDataset(torch.utils.data.Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, fold=0, split='train', samples_per_epoch=100, device='cpu'):
       
#         with open('data/train_data.pickle', 'rb') as f:
#             X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        
#         X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.

#         self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
#         self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
#         self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)

#         # split train set in train/val
#         train_mask = train_mask_[:,fold]
#         nval = int(train_mask.sum()*0.2)
#         val_idxs = np.random.RandomState(1).choice(np.nonzero(train_mask.flatten())[0],(nval,),replace=False)
#         train_mask[val_idxs] = 0;
#         val_mask = train_mask*0
#         val_mask[val_idxs] = 1
                          
#         print('DATA STATS: train: %d val: %d' % (train_mask.sum(),val_mask.sum()))
            
#         if split=='train':
#             self.mask = torch.from_numpy(train_mask).to(device)
#         if split=='val':
#             self.mask = torch.from_numpy(val_mask).to(device)
#         if split=='test':
#             self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
#         self.samples_per_epoch = samples_per_epoch

#     def __len__(self):
#         return self.samples_per_epoch

#     def __getitem__(self, idx):
#         return self.X,self.y,self.mask
    


def get_planetoid_dataset(name, normalize_features=True, transform=None, split="complete"):
    path = osp.join('.', 'data', name)
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, name, split=split)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y,dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        
        if split=='train':
            self.mask = dataset[0].train_mask.to(device)
        if split=='val':
            self.mask = dataset[0].val_mask.to(device)
        if split=='test':
            self.mask = dataset[0].test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.edge_index


class CarpetDataset(torch.utils.data.Dataset):
    _df: pd.DataFrame = None
    _val_indices = _test_indices = list()

    def __init__(self, 
            filename = 'carpet.csv', 
            split=DataSeparation.TRAIN, 
            device='cpu', 
            use_unlabeled: bool = False,
            label_col: str = 'class',
            use_non_bio_features: bool = False, 
            use_bio_features: bool = False,
            use_carpet_features: bool = True,
            shuffle_seed: int = 12345,
            val_test_split: float = 0.1,
            fill_na_vals: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df.fillna(df.mean(axis=0)),
            samples_per_epoch: int = 100) -> None:

        self.samples_per_epoch: int = samples_per_epoch
        self.device = device
        
        self._use_unlabeled = use_unlabeled
        self._label_col = label_col
        self._use_non_bio_features = use_non_bio_features
        self._use_bio_features = use_bio_features
        self._use_carpet_features = use_carpet_features
        self._shuffle_seed = shuffle_seed
        self._val_test_split= val_test_split
        self._fill_na_vals = fill_na_vals

        if CarpetDataset._df is None:
            self._load_and_preprocess(filename)
            self._split()
        self.num_classes = len(self._df[self._label_col].unique().tolist())

        self.mask = np.zeros(self._df.shape[0], dtype=bool)
        if split == DataSeparation.VAL:
            self.mask[self._val_indices] = True
        elif split == DataSeparation.TEST:
            self.mask[self._test_indices] = True
        else:
            self.mask[self._val_indices + self._test_indices] = True
            self.mask = ~self.mask

        # Convert to numeric value the labels
        codes, uniques = pd.factorize(self._df[label_col])
        encoded_labels = {i: el for i, el in enumerate(uniques)}
        self.y = one_hot_embedding(codes.tolist(), self.num_classes).to(device)
        
        self.X = torch.from_numpy(self._df.drop(self._label_col, axis=1).values.astype(np.float32)).to(device)
        self.mask = torch.tensor(self.mask).to(device)
        
        self.n_features = self.X.size(1)

        print(f"\n@@@ Data Processed: X shape {self.X.shape}, y shape {self.y.shape} @@@", flush=True)
        print(f"\n@@@ Encoded labels: {encoded_labels} @@@", flush=True)


    def _load_and_preprocess(self, filename):
        df = pd.read_csv(f"data/raw/{filename}", header=0)

        # Drop unlabeled samples if necessary
        if not self._use_unlabeled:
            df = df[df[self._label_col] != self.unlabeled]

        # Shuffle DataFrame
        df = df.sample(frac=1, random_state=self._shuffle_seed)

        print(f"Raw DataFrame Shape: {df.shape}", flush=True)
        
        # Prune features if necessary
        if not self._use_non_bio_features:
            df.drop(self.non_bio_cols, axis=1, inplace=True)
        if not self._use_bio_features:
            df.drop(self.bio_cols, axis=1, inplace=True)
        if not self._use_carpet_features:
            df = df.loc[:, ~df.columns.str.startswith(self.carpet_features_start_str)]

        print(f"\n@@@ DataFrame Shape after Pruning: {df.shape}, use_non_bio_features? {self._use_non_bio_features}, \
            use_bio_features? {self._use_bio_features}, use_carpet_features? {self._use_carpet_features} @@@", flush=True)

        # Fill NaN values
        df = self._fill_na_vals(df)
        print(f"\n@@@ NaN Status: {len(df.isnull().sum(axis = 0).to_numpy().nonzero())} out of {df.shape[1]} cols have at least one NaN values.", flush=True)

        # Normalize: use min-max
        labels = df[self._label_col]
        df.drop(self._label_col, axis=1, inplace=True)
        df = (df - df.min()) / (df.max() - df.min() + 1e-6)

        df[self._label_col] = labels

        CarpetDataset._df = df

    def _split(self) -> None:
        # Split to train/val/test
        df = CarpetDataset._df
        labels = df[self._label_col].unique().tolist()

        for label in labels:
            inds = df.index[df[self._label_col] == label].tolist()
            l = len(inds)
            val_size = int(l * self._val_test_split)
            train_size = l - 2 * val_size
            arr_inds = np.split(inds, [train_size, train_size + val_size])
            CarpetDataset._val_indices = CarpetDataset._val_indices + list(arr_inds[1])
            CarpetDataset._test_indices = CarpetDataset._test_indices + list(arr_inds[2])
    
    @property
    def non_bio_cols(self):
        return ['metaclass', 'stage', 'fallstatus', 'fallseverity', 'birthdate', 'visit', 'subjectID', 'sessionID']
    
    @property
    def bio_cols(self):
        return ['gender', 'age', 'height', 'weight', 'leglengthL', 'leglengthR']

    @property
    def carpet_features_start_str(self):
        return ('PS_', 'SS_', 'MS_', 'HR_', 'EC_', 'DTC_', 'DTS_', 'DTM_')

    @property
    def unlabeled(self):
        return 'unlabeled'

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        self_loop_edge_index = torch.repeat_interleave(torch.arange(self.X.size(0)).unsqueeze(0), 2, dim=0)
        return self.X,self.y,self.mask,self_loop_edge_index