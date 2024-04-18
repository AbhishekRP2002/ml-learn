import numpy as np 
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')
logger.setLevel(logging.INFO)


class Perceptron:
    """
    Class to represent perceptron model
    """
    
    def __init__(self,learning_rate = 0.01, n_iterations = 1000):
        """
        Constructor for perceptron model
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def initialize(self, n_features):
        """
        Method to initialize weights and bias
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
        return 
    
    def activation_function(self, x):
        """
        Method to define the activation function     
        """
        # TODO: Now it is a simple step function, but it can be replaced by other functions -> sigmoid or tanh or relu as per requirement
        return 1 if x > 0 else 0
    
    def predict(self , input):
        """
        Method to predict new class labels for input data
        """
        pre_activation_value = np.dot(self.weights, input) + self.bias
        return self.activation_function(pre_activation_value)
    
    # Train the perceptron model with Gradient Descent algorithm
    def train(self, X, Y):
        self.initialize(X.shape[1])
        for _ in range(self.n_iterations):
            for input, label in zip(X, Y):
                y_pred = self.predict(input)
                error = label - y_pred # Note : y_pred - label nhi hoga 
                self.weights += self.learning_rate * error * input
                self.bias += self.learning_rate * error
        return
    

if __name__ == "__main__":

    # Test the perceptron model
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 0, 0, 1])
    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.train(X, Y)
    logger.info(f"Predictions: {[perceptron.predict(x) for x in X]}")
    logger.info(f"Weights: {perceptron.weights} -- Bias: {perceptron.bias}")
