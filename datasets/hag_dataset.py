import csv
import os
import json
import numpy as np
import random
import time
import pickle
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


def open_csv_per_line(file_path, is_header = True):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        if is_header: 
            next(reader)
        yield from reader


def load_pickle_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class HAGDataset(Dataset):
    def __init__(self, cfg, split, img_feat_base_path):
        self.cfg = cfg
        self.img_feat_base_path = Path(img_feat_base_path)

        print('Speaker Dataset loading vocab json file: ', cfg.data.vocab_json)
        self.vocab_json = cfg.data.vocab_json
        self.word_to_idx = json.load(open(self.vocab_json, 'r'))
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word

        self.vocab_size = len(self.idx_to_word)
        print('vocab size is ', self.vocab_size)

        self.split = split

        split_pickle_path = "hag_data/{}_data.pickle".format(split)
        self.split_data = load_pickle_data(split_pickle_path)
        self.split_data = [tmp for tmp in self.split_data if len(tmp) == 4]

        self.num_samples = len(self.split_data)

        if split == 'train':
            self.batch_size = cfg.data.train.batch_size
            self.seq_per_img = cfg.data.train.seq_per_img
        elif split == 'dev': 
            self.batch_size = cfg.data.val.batch_size
            self.seq_per_img = cfg.data.val.seq_per_img
        elif split == 'test': 
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
        else:
            raise Exception('Unknown data split %s' % split)

        seq_size = len(self.split_data[0][-1])
        self.max_seq_length = seq_size
        print('Max sequence length is %d' % self.max_seq_length)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        random.seed()

        idx, img_1_img_path, img_2_img_path, cap_data = self.split_data[i]

        img_1_base_path = Path(img_1_img_path).name
        img_2_base_path = Path(img_2_img_path).name

        img_1_npy_path = str(self.img_feat_base_path / (img_1_base_path + ".npy"))
        img_2_npy_path = str(self.img_feat_base_path / (img_2_base_path + ".npy"))

        img_1_feature = torch.from_numpy(np.load(img_1_npy_path))
        img_2_feature = torch.from_numpy(np.load(img_2_npy_path))

        cap_data = torch.from_numpy(np.array(cap_data, dtype=np.int))
        mask = (cap_data != 0)

        return img_1_feature, img_2_feature, cap_data, mask, \
            img_1_img_path, img_2_img_path

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length


#def hag_collate(batch):
    #return batch

#class HAGDataLoader(DataLoader):
    
    #def __init__(self, dataset, **kwargs):
        #kwargs['collate_fn'] = hag_collate
        #super().__init__(dataset, **kwargs)
