import pandas as pd
import json
import os

# Paths
save_path = "/work/hdd/bdta/aqian1/distillEBWM/data/dolly/"
os.makedirs(save_path, exist_ok=True)

# Read train and valid CSV files
train_df = pd.read_csv(os.path.join(save_path, "train.csv"))
valid_df = pd.read_csv(os.path.join(save_path, "valid.csv"))

# Replace NaN in 'context' with an empty string
train_df['context'] = train_df['context'].fillna("")
valid_df['context'] = valid_df['context'].fillna("")

# Combine train and validation into a single raw.jsonl file
raw_data = pd.concat([train_df, valid_df], ignore_index=True)

with open(os.path.join(save_path, "raw.jsonl"), "w") as f:
    for _, row in raw_data.iterrows():
        json.dump({"instruction": row["instruction"], "input": row["context"], "output": row["response"]}, f)
        f.write("\n")

print(f"Raw dataset saved to {os.path.join(save_path, 'raw.jsonl')}")