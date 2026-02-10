#!/usr/bin/env python3
"""
Simple script to check training dataset structure
"""

import pandas as pd
import os

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
print(df_train.head(3).to_string())

# Check the structure of the data
print("\n" + "="*60)
print("SAMPLE PATHS ANALYSIS:")
for i in range(min(3, len(df_train))):
    row = df_train.iloc[i]
    print(f"\nRow {i}:")
    for col in df_train.columns:
        if 'image' in col.lower() or 'path' in col.lower() or 'file' in col.lower():
            value = row[col]
            exists = os.path.exists(str(value)) if pd.notna(value) else False
            print(f"  {col}: {value} [EXISTS: {exists}]")

print("\n" + "="*60)
print("ALL COLUMNS SAMPLE:")
if len(df_train) > 0:
    row = df_train.iloc[0]
    for col in df_train.columns:
        print(f"{col}: {row[col]}")

print("\nDataset analysis complete!")