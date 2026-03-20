import pandas as pd
import numpy as np
import glob
import os

# DATA LOADING

# Combine all 8 CSVs from the 'dataset' folder into a single DataFrame
path = r'./dataset' 
all_files = glob.glob(os.path.join(path, "*.csv"))

df_list = []
for file in all_files:
    print(f"Loading {os.path.basename(file)}...")
    df_temp = pd.read_csv(file)
    df_list.append(df_temp)

# Combine them all into one DataFrame
df = pd.concat(df_list, ignore_index=True)
print(f"Total rows loaded: {len(df)}")


### PREPROCESSING

## Clean Headers and Labels
# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Strip spaces from the target labels to prevent duplicate classes 
# (e.g., 'Web Attack ' vs 'Web Attack')
df['Label'] = df['Label'].str.strip()


## Handle Infinite Values and NaNs
import numpy as np

print(f"Initial shape: {df.shape}")

# Replace positive and negative infinity with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows that contain NaN values
df.dropna(inplace=True)

print(f"Shape after removing Infs/NaNs: {df.shape}")

# Identify columns where all values are the same (variance = 0)
# (Excluding the 'Label' column)
numeric_df = df.drop(columns=['Label'])
constant_columns = numeric_df.columns[numeric_df.nunique() <= 1]

print(f"Dropping {len(constant_columns)} constant columns: {list(constant_columns)}")
df.drop(columns=constant_columns, inplace=True)


## Label Encoding
from sklearn.preprocessing import LabelEncoder

# Separate features (X) and target (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Convert text labels to integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Print the mapping so you know which number corresponds to which attack later
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Label Mapping:", label_mapping)


## Train-Test Split
from sklearn.model_selection import train_test_split

# Split 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded # Stratify to maintain class distribution in train and test sets
)

## Calculate Sample Weights for Imbalanced Classes
from sklearn.utils.class_weight import compute_sample_weight

print("Calculating sample weights...")
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)