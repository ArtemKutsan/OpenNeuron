import numpy as np


# Класс для оптимизаторов
class Optimizer:
    def update(self, object):
        raise NotImplementedError('This method should be overridden by subclasses')

class SGD(Optimizer):
    def __init__(self, learning_rate=0.1, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.m = {'weights': {}, 'bias': {}}

    def update(self, object):
        # Метод инициализации массива delta оптимизатора для текущего объекта, если еще не инициализирован
        self.m['weights'].setdefault(object.id, np.zeros_like(object.weights))
        self.m['bias'].setdefault(object.id, np.zeros_like(object.bias))
        # print(f'object.weights:\n{object.weights}')
        # print(f'object.bias: {object.bias}')
        m_weights = self.learning_rate * object.gradient['weights'] + self.momentum * self.m['weights'][object.id]
        object.weights -= m_weights
        self.m['weights'][object.id] = m_weights

        m_bias = self.learning_rate * object.gradient['bias'] + self.momentum * self.m['bias'][object.id]
        not_none = np.where(object.bias != None)
        object.bias[not_none] -= m_bias[not_none]
        self.m['bias'][object.id] = m_bias

    def __str__(self):
        return f'"SGD"' 

class Adam(Optimizer):
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, initial_step=1, gradient_threshold=False):
        self.m = {'weights': {}, 'bias': {}}
        self.v = {'weights': {}, 'bias': {}}
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = 1e-8
        self.t = 0
        self.learning_rate = learning_rate
        self.step = initial_step
        self.gradient_threshold = gradient_threshold

    def update(self, object):
        # Initialize moments if they do not exist
        self.m['weights'].setdefault(object.id, np.zeros_like(object.weights))
        self.v['weights'].setdefault(object.id, np.zeros_like(object.weights))
        self.m['bias'].setdefault(object.id, np.zeros_like(object.bias))
        self.v['bias'].setdefault(object.id, np.zeros_like(object.bias))

        self.t += 1
        if self.gradient_threshold:
            gradient_norm = np.linalg.norm(object.gradient['weights']) + np.linalg.norm(object.gradient['bias'])
            if gradient_norm > self.gradient_threshold:
                self.t += self.step
        
        corrected_learning_rate = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        # Update weights
        self.m['weights'][object.id] = self.beta1 * self.m['weights'][object.id] + (1 - self.beta1) * object.gradient['weights']
        self.v['weights'][object.id] = self.beta2 * self.v['weights'][object.id] + (1 - self.beta2) * (object.gradient['weights'] ** 2)
        m_weights_hat = self.m['weights'][object.id] / (1 - self.beta1 ** self.t)
        v_weights_hat = self.v['weights'][object.id] / (1 - self.beta2 ** self.t)

        object.weights -= corrected_learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)

        # Update bias
        self.m['bias'][object.id] = self.beta1 * self.m['bias'][object.id] + (1 - self.beta1) * object.gradient['bias']
        self.v['bias'][object.id] = self.beta2 * self.v['bias'][object.id] + (1 - self.beta2) * (object.gradient['bias'] ** 2)
        m_bias_hat = self.m['bias'][object.id] / (1 - self.beta1 ** self.t)
        v_bias_hat = self.v['bias'][object.id] / (1 - self.beta2 ** self.t)

        not_none = np.where(object.bias != None)
        object.bias[not_none] -= corrected_learning_rate * m_bias_hat[not_none] / (np.sqrt(v_bias_hat[not_none]) + self.epsilon)

    def __str__(self):
        return '"Adam"'