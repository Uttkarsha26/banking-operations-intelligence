
from pathlib import Path

import pandas as pd

# Find the project folder automatically.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load the training and test datasets.
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Display basic information about the datasets.
print("BANKING77 DATA INSPECTION")
print("-" * 40)

print(f"Training rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nFirst five training examples:")
print(train_df.head())

print("\nNumber of training categories:")
print(train_df["category"].nunique())

print("\nTop 10 training categories:")
print(train_df["category"].value_counts().head(10))

print("\nMissing values in training data:")
print(train_df.isnull().sum())

print("\nDataset inspection completed!")
