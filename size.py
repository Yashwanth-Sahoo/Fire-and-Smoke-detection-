import glob
import os

# The folder structure visible in your sidebar
base_path = "dataset/data"

print("==============================")
print("THE SCRIPT IS RUNNING!")
print("==============================")

train_files = glob.glob(f"{base_path}/train/images/*")
val_files = glob.glob(f"{base_path}/val/images/*")
test_files = glob.glob(f"{base_path}/test/images/*")

print(f"Total Train Images: {len(train_files)}")
print(f"Total Val Images:   {len(val_files)}")
print(f"Total Test Images:  {len(test_files)}")