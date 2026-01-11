'''
Dataset/sample_generator.py

This script is used to produce random tiny samples from the original dataset.
These samples are used for testing the app.
'''

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
rng = np.random.default_rng(seed=12349)

data_path: str = os.getenv("DATA_PATH")
df_main: pd.DataFrame = pd.read_csv(data_path)
print(f"Original shape: {df_main.shape}")

df_sampled: pd.DataFrame = df_main.sample(n=80)

df_return: pd.DataFrame = df_sampled.drop(columns=["id","Class"])

def add_minus(x):
  val = rng.uniform(low=-2.0,high=2.0)
  return x+val

df_return = df_return.map(add_minus)

print(f"Sampled shape: {df_return.shape}")

df_return.to_csv("Dataset/sample_data.csv",index=False)