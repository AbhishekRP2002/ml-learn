import numpy as np

class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        cov_matrix = np.cov(X, rowvar=False)
        eigen_values, self.eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigen_values)[::-1]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
    
    def transform(self, X):
        normalized_X = X - self.mean
        reduced_X = np.dot(normalized_X, self.eigenvectors[:, self.n_components])
        return reduced_X
    
    def inverse_transform(self, reduced_X):
        reconstructed_X = np.dot(reduced_X, self.eigenvectors[:, :self.n_components].T) + self.mean
        return reconstructed_X
        
        
if __name__ == "__main__":
    pass
