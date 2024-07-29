import numpy as np
import tensorflow as tf

from openneuron.utils import format


# Класс для нейрона
class Neuron:
    count = 0
    def __init__(self, weights, bias=None, activation=None):
        self.index = Neuron.count  # индекс нейрона в слое
        self.layer = None  # ссылка на слой будет установлена позже
        # self.number = None
        self.init_Inputs = None
        self.init_weights = np.array(weights, dtype=np.float32)  # начальные значения весов
        self.init_bias = None if bias is None else np.array(bias, dtype=np.float64)  # начальные значения смещений
        self.activation = activation

    @property
    def Inputs(self):
        if self.layer is not None:
            return tf.keras.backend.eval(self.layer.Inputs)
        return self.init_Inputs
    
    @Inputs.setter
    def Inputs(self, value):
        self.init_Inputs = value
    
    @property
    def inputs(self):
        return self.Inputs[-1] if self.Inputs is not None else None
    
    @property
    def weights(self):
        if self.layer is None:
            return self.init_weights
        return self.layer.kernel[:, self.index]  # получаем значение весов из матрицы весов слоя

    @weights.setter
    def weights(self, value):
        if self.layer is not None:
            self.layer.kernel[:, self.index].assign(value)  # обновляем значения весов в матрице весов слоя

    @property
    def bias(self):
        if self.init_bias is None:
            return None
        return self.layer.bias[self.index].numpy() if self.layer is not None else self.init_bias  # получаем значение смещения из матрицы смещений слоя

    @bias.setter
    def bias(self, value):
        if self.layer is not None and self.init_bias is not None:
            self.layer.bias[self.index].assign(value)  # обновляем значения смещения в матрице смещений слоя

    @property
    def Z(self):
        if self.layer is not None:
            return tf.keras.backend.eval(self.layer.Z)[-1]
        return (np.dot(self.Inputs, self.weights) + (self.bias or 0)) if self.Inputs is not None else None
    
    @property
    def z(self):
        return self.Z[self.index] if self.Z is not None else None
    
    @property
    def A(self):
        if self.layer is not None:
            return tf.keras.backend.eval(self.layer.A)[-1]
        return self.activation(self.Z).numpy() if self.Z is not None else None
    
    @property
    def a(self):
        return self.A[self.index] if self.A is not None else None
    
    def call(self, Inputs):
        self.Inputs = Inputs
        return self.A
    
    def __call__(self, Inputs):
        return self.call(Inputs)

    def __str__(self):
        if self.bias is not None:
            return f'Neuron {self.index + 1}, inputs: {self.inputs}, weights: {self.weights}, bias: {self.bias:.4f}, ' \
                   f'z: {self.z:.4f}, activation: {self.activation.__name__ if self.activation is not None else "None"}, '\
                   f'a: {self.a:4f}'
        else:
            return f'Neuron {self.index + 1}, inputs: {self.inputs}, weights: {self.weights}, z: {self.z:.4f}, ' \
                   f'activation: {self.activation.__name__ if self.activation is not None else "None"}, a: {self.a:.4f}'