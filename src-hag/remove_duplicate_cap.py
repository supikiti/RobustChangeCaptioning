import csv
from pathlib import Path


def open_csv_per_line(file_path, is_header = True):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        if is_header: 
            next(reader)
        yield from reader


def write_csv(data, path):
    with open(path, "w") as f:
        for line in data:
            tmp = ",".join(line) + "\n"
            f.write(tmp)


path_to_img_cap_pair = "hag_data/img_cap_pair.txt"
path_to_train_saved_csv = "hag_data/train_img_cap_pair.txt"
path_to_dev_saved_csv = "hag_data/dev_img_cap_pair.txt"
path_to_eval_saved_csv = "hag_data/eval_img_cap_pair.txt"

img_set = set()
new_data = []
csv_generator = open_csv_per_line(path_to_img_cap_pair)
for per_line in csv_generator:
    if per_line[0] not in img_set:
        img_set.add(per_line[0])
        new_data.append(per_line)

write_csv(new_data[:-20], path_to_train_saved_csv)
write_csv(new_data[-20:-10], path_to_dev_saved_csv)
write_csv(new_data[-10:], path_to_eval_saved_csv)