import numpy as np


# Классы для функции потерь
class LossFunction:
    def evaluate_error(self, y, predictions):
        raise NotImplementedError('This method should be overridden by subclasses')
    
    def loss_derivative(self, error):
        raise NotImplementedError('This method should be overridden by subclasses')

    def evaluate_loss(self, y, predictions):
        raise NotImplementedError('This method should be overridden by subclasses')

class MAE(LossFunction):
    def evaluate_error(self, y, predictions):
        return (predictions - y) / len(y)
    
    def loss_derivative(self, error):
        return np.sign(error)
    
    def evaluate_loss(self, y, predictions):
        return np.mean(np.abs(predictions - y))
    
    def __str__(self):
        return '"Mean Absolute Error"'
    
class MSE(LossFunction):
    def evaluate_error(self, y, predictions):
        return (predictions - y) / len(y)
    
    def loss_derivative(self, error):
        return 2 * error
    
    def evaluate_loss(self, y, predictions):
        return np.mean(np.square(predictions - y))

    def __str__(self):
        return '"Mean Squared Error"'

class CE(LossFunction):
    def evaluate_error(self, y, predictions):
        return predictions - y
    
    def loss_derivative(self, error):
        return error

    def evaluate_loss(self, y, predictions):
        epsilon = 1e-8
        return -np.mean(np.sum(y * np.log(predictions + epsilon), axis=1))
    
    def __str__(self):
        return '"Categorical Crossentropy"'