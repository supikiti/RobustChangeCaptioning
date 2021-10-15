import argparse
import csv
from pathlib import Path


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap_res_dir_path', type=str, default = "data/cap_results")
    parser.add_argument("--question_path", default = "data/all_img_csv")
    parser.add_argument("--save_txt_file_path", default = "data/img_cap_pair.txt")
    args = parser.parse_args()
    return args


def open_csv_per_line(file_path, is_header = True):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        if is_header: 
            next(reader)
        yield from reader


def get_img_name_dict_from_path_list(question_file_list):
    img_name_dict = dict()
    for file_path_i in question_file_list:
        csv_generator = open_csv_per_line(file_path_i, is_header = True)
        for line in csv_generator:
            index = line[0]
            data = line[2:]
            img_name_dict[index] = data

    return img_name_dict


def generator_from_lists_per_num(list_1, list_2, num = 3):
    assert len(list_1) % num == 0
    assert len(list_2) % num == 0

    for i in range(0, len(list_1) - num, num):
        yield list_1[i : i + num], list_2[i : i + num]
        

def trim_img_file_name(img_name):
    return str(Path(img_name).stem)
    

def get_img_cap_pair_list(cap_res_file_list, img_name_dict):
    img_cap_pair_list = []
    for file_path_i in cap_res_file_list:
        line_generator = open_csv_per_line(file_path_i, is_header = True)
        for line in line_generator:
            index = line[4]
            res_data = line[5:]
            que_data = img_name_dict[index]

            for que_i, res_i in generator_from_lists_per_num(que_data, res_data):
                if res_i[0] == "2":
                    continue
                else:
                    img_1, img_2 = trim_img_file_name(que_i[0]), trim_img_file_name(que_i[1])
                    cap = res_i[-1]
                    img_cap_pair_list.append([img_1, img_2, cap])
                
    return img_cap_pair_list


def save_list_as_txt_file(data_list, path):
    with open(path, "w") as f:
        for list_i in data_list:
            line = ",".join(list_i) + "\n"
            f.write(line)
            

if __name__ == "__main__":
    args = arg_parse()
    
    question_file_list = list(Path(args.question_path).glob("*.csv"))
    img_name_dict = get_img_name_dict_from_path_list(question_file_list)
    
    cap_res_file_list = list(Path(args.cap_res_dir_path).glob("*.csv"))
    img_cap_pair_list = get_img_cap_pair_list(cap_res_file_list, img_name_dict)

    save_list_as_txt_file(img_cap_pair_list, args.save_txt_file_path)