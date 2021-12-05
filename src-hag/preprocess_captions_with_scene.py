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
from utils.preprocess import tokenize_graph, tokenize_jp, encode_jp, build_vocab_jp
from utils.preprocess import encode_graph, tokenize_graph


def parser():
    # Load config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_cap_pair_file_path", 
        type = str, \
        default = "/mnt/home/taiki-n/riken/data/home-action-genome/true_res/new_true_dataset_with_scene.csv"
        )
    parser.add_argument(
        '--input_vocab_json', 
        default=None
        )
    parser.add_argument(
        '--output_vocab_json', 
        default='hag_data_with_scene/vocab.json', 
        help='output vocab file'
        )
    parser.add_argument(
        '--output_scene_json', 
        default='hag_data_with_scene/scene.json', 
        help='output scene dict'
        )
    parser.add_argument(
        '--output_pickle_dir', 
        default="hag_data_with_scene", 
        help='output h5 file'
        )
    parser.add_argument(
        '--word_count_threshold', 
        default=1, 
        type=int
        )
    parser.add_argument(
        '--allow_unk', 
        default=0, 
        type=int
        )

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


def get_cap_part_from_dict(idx_to_img_pair_gen):
    for *other, cap in idx_to_img_pair_gen:
        yield [other, data[-1]]


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


def tokenize_generator(data, tokenizer=None):
    for *others, cap, graph_path in data:
        cap_tokens = tokenize_jp(cap,
                                add_start_token=True,
                                add_end_token=True,
                                punct_to_remove=[",", "，", "、", ".", "。", "．"],
                                tokenizer=tokenizer)

        graph_data = load_pickle_data(graph_path)
        graph_data = tokenize_graph(
            graph_data,
            add_start_token=True,
            add_end_token=True
        )
        
        yield [*others, cap_tokens, graph_data]


def get_max_vocab_length(data_list):
    max_length = -1
    for *othres, cap_token, _ in data_list:
        if max_length < len(cap_token):
            max_length = len(cap_token) 
    return max_length


def get_max_scene_length(data_list):
    max_length = -1
    for *others, graph_data in data_list:
        if max_length < len(graph_data):
            max_length = len(graph_data)
    return max_length


def get_encoded_data(data_list, word_to_idx, scene_to_idx, 
        max_vocab_length, max_scene_length):
    data_list = sorted(data_list, key=lambda x:int(x[0]))

    for *others, tokens_list, graph_data in data_list:
        # encode vocab list
        Li = np.zeros(max_vocab_length, dtype=np.int)
        tokens_encoded = encode_jp(
            tokens_list,
            word_to_idx,
            allow_unk=args.allow_unk == 1
        )

        for k, w in enumerate(tokens_encoded):
            Li[k] = w

        # encode scene graph list
        Si = np.zeros(max_scene_length, dtype=np.int)
        graph_encoded = encode_graph(
            graph_data, 
            scene_to_idx,
            allow_unk=args.allow_unk == 1
        )

        for k, w in enumerate(graph_encoded):
            Si[k] = w

        yield [*others, list(Li), list(Si)]


def save_data_as_pickle(saved_data, path):
    with open(path, "wb") as f:
        pickle.dump(saved_data, f)


def load_pickle_data(path):
    with open(path, "rb") as f:
        saved_data = pickle.load(f)
    return saved_data


def get_word_to_idx_dict(all_data, input_vocab_json):
    ## Either create the vocab or load it from disk
    if args.input_vocab_json is None:
        word_to_idx = build_vocab_jp(
            all_data,
            min_token_count=args.word_count_threshold,
            punct_to_remove=[",", "，", "、", ".", "。", "．"]
        )
    else:
        print('Loading vocab')
        with open(args.input_vocab_json, 'r') as f:
            word_to_idx = json.load(f)

    return word_to_idx

def save_json_file(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def get_scene_to_idx_dict(all_data):
    obj_set, graph_set = set(), set()
    for *_, graph_path in all_data:
        graph_data = load_pickle_data(graph_path)
        for graph_i in graph_data:
            obj_set.add(graph_i[0])
            graph_set = graph_set | set(graph_i[1])
    
    obj_list = sorted(list(obj_set))
    graph_list = sorted(list(graph_set))
    all_list = obj_list + graph_list

    scene_to_idx_dict = {
        '<NULL>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
    }

    for obj_graph_i in all_list:
        scene_to_idx_dict[obj_graph_i] = len(scene_to_idx_dict)
    return scene_to_idx_dict


def is_file_exists(path):
    return Path(path).exists()


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def main(args):
    all_data, train_data, dev_data, eval_data = \
        get_list_and_cap_data_from_path(args.img_cap_pair_file_path)

    if not is_file_exists(args.output_vocab_json):
        print("BUILDING VOCAB FILE" )
        word_to_idx = get_word_to_idx_dict(all_data, args.input_vocab_json)
        save_json_file(args.output_vocab_json, word_to_idx)
    else:
        print("LOADING VOCAB JSON FILE")
        word_to_idx = load_json_file(args.output_vocab_json)

    if not is_file_exists(args.output_scene_json):
        print("BUILDING SCENE JSON FILE" )
        scene_to_idx = get_scene_to_idx_dict(all_data)
        save_json_file(args.output_scene_json, scene_to_idx)
    else:
        print("LOADING SCENE JSON FILE")
        scene_to_idx = load_json_file(args.output_scene_json)

    # Encode all captions
    # First, figure out max length of captions
    t = Tokenizer()

    all_cap_tokens = list(tokenize_generator(all_data, t))
    train_cap_tokens = list(tokenize_generator(train_data, t))
    dev_cap_tokens = list(tokenize_generator(dev_data, t))
    eval_cap_tokens = list(tokenize_generator(eval_data, t))

    max_vocab_length = get_max_vocab_length(train_cap_tokens)
    max_scene_length = get_max_scene_length(train_cap_tokens)

    #all_cap_encoded = get_encoded_data(all_cap_tokens, word_to_idx)
    train_encoded = list(get_encoded_data(train_cap_tokens, word_to_idx, scene_to_idx, max_vocab_length, max_scene_length))
    dev_encoded = list(get_encoded_data(dev_cap_tokens, word_to_idx, scene_to_idx, max_vocab_length, max_scene_length))
    eval_encoded = list(get_encoded_data(eval_cap_tokens, word_to_idx, scene_to_idx, max_vocab_length, max_scene_length))

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
