import numpy as np 
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import sklearn.datasets as sk_data
from sklearn.pipeline import make_pipeline, Pipeline
import sklearn.preprocessing as sk_preprocess
import sklearn.linear_model as sk_linearmodel
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_metrics


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")
logger.setLevel(logging.INFO)
pd.set_option('display.max_colwidth', None)

# Define the problem statement / goal / objective (OKRs):
# - Understand how to write modular code in ML with best practices in mind
# - Understand how to use sklearn and create generic templates for model building and inference
# - Use the heart.csv data for classification task

def load_data(filename):
    try:
        data = pd.read_csv(filename, header = 0)
        logger.info(f"{data.head()}")
        logger.info(f"Data loaded from {filename}")
    except Exception as e:
        logger.error(f"Error loading data from {filename} - {e}")
    return data

#After loading let's understand the data
def show_data_properties(df:pd.DataFrame) -> None:
    df_columns = df.columns
    df_summary = df.info()
    df_shape = df.shape
    df_descriptive_stats = df.describe()
    null_status = df.isnull().sum()
    logger.info(f"Columns: \n{df_columns}")
    logger.info(f"Summary: \n{df_summary}")
    logger.info(f"Shape of the dataframe: \n{df_shape}")
    logger.info(f"Descriptive Stats: \n{df_descriptive_stats}")
    logger.info(f"Null Status: \n{null_status}") # No null values in any of the features
    return

def split_data(df:pd.DataFrame):
    X = df.drop("output", axis=1)
    y = df["output"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_run_pipeline(X_train,y_train, X_test):
    pipeline = make_pipeline(sk_preprocess.StandardScaler(), sk_linearmodel.LogisticRegression())
    pipeline.fit(X_train, y_train)
    scaler = sk_preprocess.StandardScaler()
    y_pred = pipeline.predict(X_test)
    return y_pred, pipeline




if __name__ == "__main__":
    dataset = "heart.csv"
    filename = f"scikit-learn/dataset/{dataset}"
    df = load_data(filename)
    show_data_properties(df)
    # Perform the train-test split first, followed by preprocessing or data transformations to avoid data leakage or making the model biased/ highly optimistic
    X_train, X_test, y_train, y_test = split_data(df)
    # Create a pipeline for preprocessing/data transformations and model training
    model_name = "LogisticRegression"
    y_pred, pipeline = create_run_pipeline(X_train, y_train, X_test)
    accuracy = sk_metrics.accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy of the model: {accuracy}")
