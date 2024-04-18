# take two points 
import numpy as np

def sigmoid(w, x , b):
    return 1.0 / (1.0 + np.exp((-1)*(w*x + b)))

def error_function(w, x, b):
    error = 0
    for x, y in zip(X, Y):
        error += 0.5 * (y - sigmoid(w, x, b))**2
    return error 

def gradient_wrt_w(w,x, b, y):
    fx = sigmoid(w, x, b)
    return (fx - y) * fx * (1 - fx) * x

def gradient_wrt_b(w,x, b, y):
    fx = sigmoid(w, x, b)
    return (fx - y) * fx * (1 - fx)

def gradient_descent():
    w = 0.5
    b = 0.5 
    learning_rate = 0.01
    max_epochs = 10000
    for epoch in range(max_epochs):
        dw = 0
        db = 0
        # Batch Gradient Descent
        # for x, y in zip(X, Y):
        #     dw += gradient_wrt_w(w, x, b, y)
        #     db += gradient_wrt_b(w, x, b, y)
        # w = w - learning_rate * dw
        # b = b - learning_rate * db
        
        # Stochastic Gradient Descent
        for x, y in zip(X, Y):
            dw = gradient_wrt_w(w, x, b, y)
            db = gradient_wrt_b(w, x, b, y)
            w = w - learning_rate * dw
            b = b - learning_rate * db
        print("Epoch: ", epoch, "Error: ", error_function(w, x, b))
        
if __name__ == "__main__":
    X = np.array([1.0, 2.5])
    Y = np.array([2.0, 4.5])
    gradient_descent()