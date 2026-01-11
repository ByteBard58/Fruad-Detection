'''
Dataset/sample_generator.py

This script is used to produce random tiny samples from the original dataset.
These samples are used for testing the app.
'''

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

data_path: str = os.getenv("DATA_PATH")
df_main: pd.DataFrame = pd.read_csv(data_path)
print(f"Original shape: {df_main.shape}")

df_sampled: pd.DataFrame = df_main.sample(n=80)

df_return: pd.DataFrame = df_sampled.drop(columns=["id","Class"])
print(f"Sampled shape: {df_return.shape}")

df_return.to_csv("Dataset/sample_data.csv",index=False)