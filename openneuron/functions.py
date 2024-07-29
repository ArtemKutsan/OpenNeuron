import numpy as np


# Функции инициализации весов и смещений
def xavier(input_size, output_size):
    limit = np.sqrt(2. / (input_size + output_size))
    Weights = np.random.uniform(-limit, limit, (input_size, output_size))
    Biases = np.random.uniform(-limit, limit, (1, output_size))
    return Weights, Biases


# Классы для функции потерь
class LossFunction:
    def evaluate_delta(self, y, predictions):
        raise NotImplementedError('This method should be overridden by subclasses')

    def evaluate_loss(self, y, predictions):
        raise NotImplementedError('This method should be overridden by subclasses')

class MAE(LossFunction):
    def evaluate_delta(self, y, predictions):
        return np.sign(predictions - y) / len(y)
    
    def evaluate_loss(self, y, predictions):
        return np.mean(np.abs(predictions - y))
    
    def __str__(self):
        return '"Mean Absolute Error"'
    
class MSE(LossFunction):
    def evaluate_delta(self, y, predictions):
        return 2 * (predictions - y) / len(y)
    
    def evaluate_loss(self, y, predictions):
        return np.mean(np.square(predictions - y))

    def __str__(self):
        return '"Mean Squared Error"'

class CE(LossFunction):
    def evaluate_delta(self, y, predictions):
        return predictions - y
    
    def evaluate_loss(self, y, predictions):
        epsilon = 1e-8
        return -np.mean(np.sum(y * np.log(predictions + epsilon), axis=1))
    
    def __str__(self):
        return '"Categorical Crossentropy"'