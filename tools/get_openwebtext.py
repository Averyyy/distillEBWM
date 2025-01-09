import datasets
import os
import re

# 指定绝对路径
BASE_DIR = "/work/hdd/bdta/aqian1/distillEBWM"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/openwebtext")

dataset = datasets.load_dataset('openwebtext', split='train')


os.makedirs(OUTPUT_DIR, exist_ok=True)

num = 0
with open(os.path.join(OUTPUT_DIR, "data.txt"), "w") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)