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


class HAGDataset_with_scene(Dataset):
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

        self.scene_json = cfg.data.scene_json
        self.scene_to_idx = json.load(open(self.scene_json, "r"))
        self.idx_to_scene = {}
        for scene, idx in self.scene_to_idx.items():
            self.idx_to_scene[idx] = scene

        self.scene_size = len(self.idx_to_scene)
        print("scene size is ", self.scene_size)
        
        split_pickle_path = "hag_data_with_scene/{}_data.pickle".format(split)
        self.split_data = load_pickle_data(split_pickle_path)
        self.split_data = [tmp for tmp in self.split_data if len(tmp) == 5]

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

        self.max_vocab_length = len(self.split_data[0][-2])
        print('Max vocab length is %d' % self.max_vocab_length)

        self.max_scene_length = len(self.split_data[0][-1])
        print('Max scene length is %d' % self.max_scene_length)

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, i):
        random.seed()

        _, img_1_img_path, img_2_img_path, cap_data, scene_data =\
             self.split_data[i]

        img_1_base_path = Path(img_1_img_path).name
        img_2_base_path = Path(img_2_img_path).name

        img_1_npy_path = str(self.img_feat_base_path / (img_1_base_path + ".npy"))
        img_2_npy_path = str(self.img_feat_base_path / (img_2_base_path + ".npy"))

        img_1_feature = torch.from_numpy(np.load(img_1_npy_path))
        img_2_feature = torch.from_numpy(np.load(img_2_npy_path))

        cap_data = torch.from_numpy(np.array(cap_data, dtype=np.int))
        vocab_mask = (cap_data != 0)

        scene_data = torch.from_numpy(np.array(scene_data, dtype=np.int))
        scene_mask = (scene_data != 0)

        return (img_1_feature, img_2_feature, cap_data, 
            scene_data, vocab_mask, scene_mask,
            img_1_img_path, img_2_img_path)

    def get_vocab_size(self):
        return self.vocab_size

    def get_scene_size(self):
        return self.scene_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_idx_to_scene(self):
        return self.idx_to_scene

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_scene_to_idx(self):
        return self.scene_to_idx

    def get_max_vocab_length(self):
        return self.max_vocab_length

    def get_max_seq_length(self):
        return self.max_scene_length
