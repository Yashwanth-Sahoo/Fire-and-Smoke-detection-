import os
import cv2
import glob
import math
import random
import warnings
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Remove the '!pip install' lines from your script. 
# You should run those directly in the IDE terminal instead:
# pip install ultralytics roboflow

warnings.filterwarnings("ignore")

# Define the local configuration string
data_yaml = """
path: /Users/yashwanth/Downloads/FIRE AND SMOKE DETECTION/dataset/data  # Keeps the path relative to your project folder
train: train/images
val: val/images
test: test/images

nc: 2
names: ['smoke', 'fire']
"""

# Save it as a configuration file in your workspace
with open("fire_smoke.yaml", "w") as f:
    f.write(data_yaml.strip())

print("--- Content of fire_smoke.yaml ---")
print(open("fire_smoke.yaml").read())