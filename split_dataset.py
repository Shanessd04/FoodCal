import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = "/Users/shanes/FoodCal/new_dataset"
output_dir = "/Users/shanes/FoodCal/data"

# Ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output directories
for split in ["train", "val", "test"]:
    Path(os.path.join(output_dir, split)).mkdir(parents=True, exist_ok=True)

# Loop through classes
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    split_data = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in split_data.items():
        split_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(split_class_dir, file)
            shutil.copy2(src, dst)

print("✅ Dataset split completed!")
