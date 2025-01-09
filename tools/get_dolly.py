from datasets import load_dataset
import os
import pandas as pd

# Load Dolly dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Define save paths
save_path = "/work/hdd/bdta/aqian1/distillEBWM/data/dolly"
os.makedirs(save_path, exist_ok=True)

# Create train-validation split (90% train, 10% validation)
train_data = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Save splits as CSV files
train_data["train"].to_pandas().to_csv(os.path.join(save_path, "train.csv"), index=False)
train_data["test"].to_pandas().to_csv(os.path.join(save_path, "valid.csv"), index=False)

print(f"Dolly dataset saved to {save_path}")