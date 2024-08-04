import numpy as np

from .activations import *
from .utils import format

# Класс для нейрона
class Neuron:
    count = 0

    def __init__(self, weights='he', bias='he', input_size=2, activation=linear):
        Neuron.count += 1
        self.number = Neuron.count  # порядковый номер объекта нейрон
        self.id = id(self)  # уникальный id объекта
        self.layer = None  # ссылка на слой в котором находится нейрон (устанавливается слоем если нейрон находится в слое)
        self.index = None  # индекс нейрона в слое
        self.weights = weights
        self.bias = bias
        self.activation = activation or linear  # функция активации нейрона
    
    @property
    # Массив значений inputs (x) батча (X)
    def Inputs(self):
        return self.layer.Inputs if self.layer is not None else self.X
    
    @Inputs.setter
    def Inputs(self, X):
        self.X = X

    @property
    # Значение inputs последнего прошедшего через нейрон объекта (в рассчетах не используется, просто для вывода информации в методе __str__)
    def inputs(self):
        return self.Inputs[-1] if self.Inputs is not None else None

    @property
    # Массив значений z батча (z каждого x в батче X)
    def Z(self):
        if self.layer is not None:
            return self.layer.Z.T[self.index]
        return (np.dot(self.Inputs, self.weights) + (self.bias or 0)) if self.Inputs is not None else None

    @property
    # Значение z последнего прошедшего через нейрон объекта (в рассчетах не используется, просто для вывода информации в методе __str__)
    def z(self):
        return self.Z.T[-1].astype(float) if self.Z is not None else None
    
    @property
    # Массив значений a батча (a каждого x в батче X)
    def A(self):
        if self.layer is not None:
            return self.layer.A.T[self.index]
        return self.activation(self.Z) if self.Z is not None else None

    @property
    # Значение a последнего прошедшего через нейрон объекта (в рассчетах не используется, просто для вывода информации в методе __str__)
    def a(self):
        return self.A.T[-1].astype(float) if self.A is not None else None

    def activation_derivative(self, Z):
        return self.activation(Z, derivative=True)

    # Метод Forward pass (прогон данных через нейрон)
    def forward(self, X, training=True):
        self.Inputs = X
        # self.Z = np.dot(self.Inputs, self.weights) + (self.bias or 0)
        # self.A = self.activation(self.Z)
        return self.A
    
    def call(self, X, training=False):
        ''' # Метод call вызывается после выполнения встроенного метода __call__.
        Служит для дополнения/изменения работы встроенного метода __call__ не нарушая необходимый для правильной работы процесс
        вычисления данных при вызове объекта. На вход получает входные данные (массив X). По умолчанию возвращает значение(я) 
        активированного выхода нейрона (массив A).
        '''
        # Любой дополнительный код
        return self.A

    def __call__(self, X, training=False):
        self.forward(X, training)
        return self.call(X, training)

    def __str__(self):
        decimals = 4
        return f'Neuron {format(self.number, decimals=decimals)}, inputs: {format(self.inputs, decimals=decimals)}, ' \
               f'weights: {format(self.weights.flatten(), decimals=decimals)}, bias: {format(self.bias.item(), decimals=decimals)}, ' \
               f'z: {format(self.z, decimals=decimals)}, activation: {self.activation.__name__}, a: {format(self.a, decimals=decimals)}'