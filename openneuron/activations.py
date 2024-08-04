import numpy as np


# Функция активации Sigmoid с расчетом производной
def sigmoid(Z, derivative=False):
    sigmoid_Z = 1 / (1 + np.exp(-Z))
    if derivative:
        return sigmoid_Z * (1 - sigmoid_Z)
    return sigmoid_Z

# Функция активации Tanh с расчетом производной
def tanh(Z, derivative=False):
    tanh_Z = np.tanh(Z)
    if derivative:
        return 1 - np.power(tanh_Z, 2)
    return tanh_Z

# Функция активации ReLU с расчетом производной
def relu(Z, derivative=False):
    if derivative:
        return (Z > 0).astype(float)
    return np.maximum(0, Z)

# Функция активации Leaky Relu с расчетом производной
def leaky_relu(Z, derivative=False):
    alpha = 0.01
    if derivative:
        return np.where(Z > 0, 1.0, alpha)
    return np.maximum(alpha * Z, Z)

# Функция активации PReLU с расчетом производной
def prelu(Z, alpha=0.01, derivative=False):
    if derivative:
        return np.where(Z > 0, 1.0, alpha)
    return np.maximum(alpha * Z, Z)

# Функция активации Gelu с расчетом производной
def gelu(Z, derivative=False):
    # Cumulative Distribution Function - кумулятивная функция распределения
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * np.power(Z, 3))))
    if derivative:
        # Probability Density Function - функция плотности вероятности
        pdf = np.exp(-0.5 * Z * Z) / np.sqrt(2 * np.pi)
        return cdf + Z * pdf
    return Z * cdf

# Функция активации ELU с расчетом производной
def elu(Z, derivative=False):
    alpha = 1.0
    if derivative:
        return np.where(Z > 0, 1, alpha * np.exp(Z))
    return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

# Функция активации SELU с расчетом производной
def selu(Z, derivative=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if derivative:
        return np.where(Z > 0, scale, scale * alpha * np.exp(Z))
    return scale * np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

# Функция активации Swish с расчетом производной
def swish(Z, derivative=False):
    sigmoid_Z = 1 / (1 + np.exp(-Z))
    if derivative:
        return sigmoid_Z + Z * sigmoid_Z * (1 - sigmoid_Z)
    return Z * sigmoid_Z

# Функция активации Softplus с расчетом производной
def softplus(Z, derivative=False):
    if derivative:
        return 1 / (1 + np.exp(-Z))  # производная Softplus это sigmoid
    return np.log(1 + np.exp(Z))

# Функция активации Softmax для преобразования в вероятности классов
def softmax(Z, derivative=False):
    if derivative:
        return np.ones_like(Z)
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # для численной стабильности
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Функция активации Linear (линейная функция)
def linear(Z, derivative=False):
    linear.__name__ = 'linear/none'
    if derivative:
        return np.ones_like(Z)
    return Z

# Функция активации Heaviside (Step Function) с расчетом производной
def heaviside(Z, threshold=0.5, derivative=False):
    if derivative:
        return np.ones_like(Z)
    return np.where(Z >= threshold, 1, 0)

'''
# Словарь с функциями активации
activation_functions = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'prelu': prelu,
    'gelu': gelu,
    'elu': elu,
    'selu': selu,
    'swish': swish,
    'softplus': softplus,
    'softmax': softmax,
    'linear': linear,
    'heaviside': heaviside,
}

# Функция для получения активационной функции по имени
def get(activation_name):
    if activation_name is None:
        return linear
    return activation_functions.get(activation_name, 'Unknown activation function name')
'''