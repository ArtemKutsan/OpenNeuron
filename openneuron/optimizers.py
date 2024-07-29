import numpy as np


# Класс для оптимизаторов
class Optimizer:
    def update(self, object):
        raise NotImplementedError('This method should be overridden by subclasses')

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.Delta = {'Weights': {}, 'Biases': {}}

    def update(self, object, learning_rate):
        # Метод инициализации массива Delta оптимизатора для текущего объекта, если еще не инициализировано
        self.Delta['Weights'].setdefault(object.id, np.zeros_like(object.Weights))
        self.Delta['Biases'].setdefault(object.id, np.zeros_like(object.Biases))

        Delta_Weights = learning_rate * object.Gradient['Weights'] + self.momentum * self.Delta['Weights'][object.id]
        object.Weights -= Delta_Weights
        self.Delta['Weights'][object.id] = Delta_Weights

        Delta_Biases = learning_rate * object.Gradient['Biases'] + self.momentum * self.Delta['Biases'][object.id]
        not_none = np.where(object.Biases != None)
        object.Biases[not_none] -= Delta_Biases[not_none]
        self.Delta['Biases'][object.id] = Delta_Biases

    def __str__(self):
        return f'"SGD" with momentum {self.momentum}' 

class Adam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, initial_step=1, gradient_threshold=False):
        self.m = {'Weights': {}, 'Biases': {}}
        self.v = {'Weights': {}, 'Biases': {}}
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = 1e-8
        self.t = 0
        self.learning_rate = learning_rate
        self.step = initial_step
        self.gradient_threshold = gradient_threshold

    def update(self, object, learning_rate):
        # Initialize moments if they do not exist
        self.m['Weights'].setdefault(object.id, np.zeros_like(object.Weights))
        self.v['Weights'].setdefault(object.id, np.zeros_like(object.Weights))
        self.m['Biases'].setdefault(object.id, np.zeros_like(object.Biases))
        self.v['Biases'].setdefault(object.id, np.zeros_like(object.Biases))

        self.t += 1
        if self.gradient_threshold:
            gradient_norm = np.linalg.norm(object.Gradient['Weights']) + np.linalg.norm(object.Gradient['Biases'])
            if gradient_norm > self.gradient_threshold:
                self.t += self.step
        
        corrected_learning_rate = learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        # Update Weights
        self.m['Weights'][object.id] = self.beta1 * self.m['Weights'][object.id] + (1 - self.beta1) * object.Gradient['Weights']
        self.v['Weights'][object.id] = self.beta2 * self.v['Weights'][object.id] + (1 - self.beta2) * (object.Gradient['Weights'] ** 2)
        m_Weights_hat = self.m['Weights'][object.id] / (1 - self.beta1 ** self.t)
        v_Weights_hat = self.v['Weights'][object.id] / (1 - self.beta2 ** self.t)

        object.Weights -= corrected_learning_rate * m_Weights_hat / (np.sqrt(v_Weights_hat) + self.epsilon)

        # Update Biases
        self.m['Biases'][object.id] = self.beta1 * self.m['Biases'][object.id] + (1 - self.beta1) * object.Gradient['Biases']
        self.v['Biases'][object.id] = self.beta2 * self.v['Biases'][object.id] + (1 - self.beta2) * (object.Gradient['Biases'] ** 2)
        m_Biases_hat = self.m['Biases'][object.id] / (1 - self.beta1 ** self.t)
        v_Biases_hat = self.v['Biases'][object.id] / (1 - self.beta2 ** self.t)

        not_none = np.where(object.Biases != None)
        object.Biases[not_none] -= corrected_learning_rate * m_Biases_hat[not_none] / (np.sqrt(v_Biases_hat[not_none]) + self.epsilon)

    def __str__(self):
        return '"Adam"'