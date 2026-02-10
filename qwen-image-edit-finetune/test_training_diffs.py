#!/usr/bin/env python3
"""
Test script to analyze training dataset differences
"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage

# Load the training dataset
train_csv = "/sc/home/felix.boelter/recreategoods/qwen-image-edit-finetune/data/example_image_dataset/train.csv"
print(f"Loading dataset from: {train_csv}")

if not os.path.exists(train_csv):
    print(f"ERROR: CSV file not found at {train_csv}")
    exit(1)

df_train = pd.read_csv(train_csv)

print(f"Training dataset: {len(df_train)} samples")
print("\nColumns:", df_train.columns.tolist())
print("\nFirst 3 samples:")
print(df_train.head(3))

# Check the structure of the data
print("\nSample paths:")
for i in range(min(3, len(df_train))):
    row = df_train.iloc[i]
    print(f"Row {i}:")
    for col in df_train.columns:
        if 'image' in col.lower() or 'path' in col.lower():
            print(f"  {col}: {row[col]}")
    print()

print("\nDataset analysis complete!")