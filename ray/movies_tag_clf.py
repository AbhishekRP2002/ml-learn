import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import ray
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')
logger.setLevel(logging.INFO)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

df = pd.read_csv("datasets/mpst_full_data.csv")
print(df.head())
# data = ray.data.read_csv("datasets/mpst_full_data.csv")
# print(data.limit(5))

columns = df.columns
logger.info(f"Columns in the dataset: {columns}")
shape = df.shape
logger.info(f"Shape of the dataset: {shape}")
