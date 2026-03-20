import pandas as pd
import numpy as np
import glob
import os

# Put all 8 CSVs in a folder named 'dataset'
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