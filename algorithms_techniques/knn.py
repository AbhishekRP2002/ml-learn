# k nearest neighbors implementation - unsupervised learning, every time for prediction, we will consider all the points in the training set and find the k nearest neighbors and then take majority vote to decide the class of the new point. 
# Define a distance metric based on which Nearest Neighbors will be found.
# Possible distance metrics: Euclidean, Manhattan, Minkowski, Hamming, Cosine

import numpy as np
import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")
logger.setLevel(logging.INFO)

class KNN:
    
    def __init__(self, k) -> None:
        self.k = k
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, distance_type = "euclidean"):
        predictions = [self._predict_for_each_instance(x) for x in X_test]
        return np.array(predictions)
    
    def calculate_distance(self, x1, x2, distance_type = "euclidean"):
        if distance_type == "cosine":
            return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return np.sqrt(np.sum((x1 - x2)**2))
        
    
    def _predict_for_each_instance(self, x, distance_type = "euclidean"):
        distances = [self.calculate_distance(x, x_train, distance_type) for x_train in self.X_train]
        top_k_indices = np.argsort(distances)[:self.k]
        top_k_labels = [self.y_train[i] for i in top_k_indices]
        most_common = np.bincount(top_k_labels).argmax()
        return most_common
    
    
if __name__ == "__main__":
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn_clf = KNN(k = 5)
    knn_clf.fit(X_train, y_train)
    predictions = knn_clf.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    logger.info(f"Accuracy: {accuracy}")
    # use cosine similarity
    predictions = knn_clf.predict(X_test, distance_type="cosine")
    accuracy = np.sum(predictions == y_test) / len(y_test)
    logger.info(f"Accuracy with cosine similarity: {accuracy}")