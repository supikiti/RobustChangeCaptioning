import csv
import pickle
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
import random
import h5py
import numpy as np

from janome.tokenizer import Tokenizer

from collections import defaultdict
from utils.preprocess import tokenize, encode, build_vocab
from utils.preprocess import tokenize_jp, encode_jp, build_vocab_jp


def parser():
    # Load config
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_cap_pair_file_path", type = str, \
        default = "/mnt/home/taiki-n/riken/data/home-action-genome/true_res/new_true_dataset_with_scene.csv")
    parser.add_argument('--input_vocab_json', default=None)
    parser.add_argument('--output_vocab_json', default='hag_data/vocab.json', help='output vocab file')
    parser.add_argument('--output_pickle_dir', default="hag_data", help='output h5 file')
    parser.add_argument('--word_count_threshold', default=1, type=int)
    parser.add_argument('--allow_unk', default=0, type=int)

    args = parser.parse_args()
    return args


def open_img_cap_pair_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        yield from reader


def split_cap_data(data_list):
    length = len(data_list)
    idxs = [int(length * num) for num in [0.85, 0.05, 0.1]]

    return_data = []
    data_len = 0
    for i, idx in enumerate(idxs):
        return_data.append(data_list[data_len : data_len + idx])
        data_len += idx

    train_list, dev_list, eval_list = return_data
    return train_list, dev_list, eval_list


def get_cap_part(data_list):
    for data_line in data_list:
        yield data_line[-1]


def get_idx_to_img_pair_data_path(img_cap_pair, img_data_base_path):
    for idx, data in enumerate(img_cap_pair):
        data_1_path = Path(img_data_base_path) / (data[0] + ".jpg.npy")
        data_2_path = Path(img_data_base_path) / (data[1] + ".jpg.npy")

        yield [idx, str(data_1_path), str(data_2_path), data[-1]]


#def get_cap_part_from_dict(idx_to_img_pair_gen):
    #for *other, cap in idx_to_img_pair_gen:
        #yield [other, cap[-1]]


def add_idx(dataset_list):
    for idx, data in enumerate(dataset_list):
        yield [idx, *data]


def get_list_and_cap_data_from_path(dataset_path):

    dataset_pair_list = list(open_img_cap_pair_csv(dataset_path))
    train_list, dev_list, eval_list = split_cap_data(dataset_pair_list)
    train_list = list(add_idx(train_list))
    dev_list = list(add_idx(dev_list))
    eval_list = list(add_idx(eval_list))

    all_idx_cap = train_list + dev_list + eval_list

    return all_idx_cap, train_list, dev_list, eval_list


def tokenize_generator(captions, tokenizer=None):
    for *others, cap, _ in captions:
        cap_tokens = tokenize_jp(cap,
                                add_start_token=True,
                                add_end_token=True,
                                punct_to_remove=[",", "，", "、", ".", "。", "．"],
                                tokenizer=tokenizer)
        yield [*others, cap_tokens]


def get_max_length(cap_tokens):
    max_length = -1
    for *othres, cap_token in cap_tokens:
        if max_length < len(cap_token):
            max_length = len(cap_token) 

    return max_length


def get_encoded_data(cap_tokens, word_to_idx):
    cap_tokens = sorted(cap_tokens, key=lambda x:int(x[0]))
    max_length = get_max_length(cap_tokens)
    
    for *others, tokens_list in cap_tokens:
        Li = np.zeros(max_length, dtype=np.int)
        tokens_encoded = encode_jp(tokens_list,
                                word_to_idx,
                                allow_unk=args.allow_unk == 1)

        for k, w in enumerate(tokens_encoded):
            Li[k] = w

        yield [*others, list(Li)]


def save_data_as_pickle(saved_data, path):
    with open(path, "wb") as f:
        pickle.dump(saved_data, f)


def load_pickle_data(path):
    with open(path, "rb") as f:
        saved_data = pickle.load(f)
    return saved_data


def main(args):
    all_data, train_data, dev_data, eval_data = \
        get_list_and_cap_data_from_path(args.img_cap_pair_file_path)

    ## Either create the vocab or load it from disk
    if args.input_vocab_json is None:
        print('Building vocab')
        word_to_idx = build_vocab_jp(
            all_data,
            min_token_count=args.word_count_threshold,
            punct_to_remove=[",", "，", "、", ".", "。", "．"]
        )
    #else:
        #print('Loading vocab')
        #with open(args.input_vocab_json, 'r') as f:
            #word_to_idx = json.load(f)

    if args.output_vocab_json is not None:
        with open(args.output_vocab_json, 'w') as f:
            json.dump(word_to_idx, f)

    # Encode all captions
    # First, figure out max length of captions
    t = Tokenizer()

    all_cap_tokens = list(tokenize_generator(all_data, t))
    train_cap_tokens = list(tokenize_generator(train_data, t))
    dev_cap_tokens = list(tokenize_generator(dev_data, t))
    eval_cap_tokens = list(tokenize_generator(eval_data, t))

    #all_cap_encoded = get_encoded_data(all_cap_tokens, word_to_idx)
    train_encoded = list(get_encoded_data(train_cap_tokens, word_to_idx))
    dev_encoded = list(get_encoded_data(dev_cap_tokens, word_to_idx))
    eval_encoded = list(get_encoded_data(eval_cap_tokens, word_to_idx))

    output_train_pickle_path = str(Path(args.output_pickle_dir) / "train_data.pickle")
    output_dev_pickle_path = str(Path(args.output_pickle_dir) / "dev_data.pickle")
    output_eval_pickle_path = str(Path(args.output_pickle_dir) / "eval_data.pickle")

    save_data_as_pickle(train_encoded, output_train_pickle_path)
    save_data_as_pickle(dev_encoded, output_dev_pickle_path)
    save_data_as_pickle(eval_encoded, output_eval_pickle_path)

    print("FINISH ENCODING !!!")
        

if __name__ == '__main__':
    args = parser()
    main(args)
