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

# This script will act as the final script that will be used to serve the model as an API using Ray Serve.
# Notes and Annotations:
# - Use Pandas for data processing, feature engineering and treat the data processing functions as tasks in Ray to leverage parallelism.Don't use Ray Data API as it has limited functionalities.
# 