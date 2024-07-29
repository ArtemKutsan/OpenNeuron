import numpy as np


# Класс для нейрона
class Neuron:
    count = 0

    def __init__(self, weights='he', bias='he', input_size=2, output_size=1, activation=None):
        Neuron.count += 1
        self.number = Neuron.count  # назначение номера нейрона
        self.id = id(self)
        self.Inputs = None  # массив значений inputs (x) батча (X)
        # self.inputs = None # значение inputs последнего прошедшего через нейрон объекта x (в рассчетах не используется, просто для вывода информации)
        self.weights = weights
        self.bias = bias
        self.Z = None  # массив значений z батча (z каждого x в батче X)
        # self.z = None  # значение z последнего прошедшего через нейрон объекта x (в рассчетах не используется, просто для вывода информации)
        self.activation = activation  # функция активации нейрона
        self.A = None  # массив значений a батча (a каждого x в батче X)
        # self.a = None  # значение a последнего прошедшего через нейрон объекта x (в рассчетах не используется, просто для вывода информации)

    @property
    # Значения inputs последнего прошедшего через нейрон объекта x (в рассчетах не используется, просто для вывода информации в методе __str__)
    def inputs(self):
        return self.Inputs[-1] if self.Inputs is not None else None

    @property
    # Значение z последнего прошедшего через нейрон объекта x (в рассчетах не используется, просто для вывода информации в методе __str__)
    def z(self):
        return self.Z[-1] if self.Z is not None else None
    
    @property
    # Значение a последнего прошедшего через нейрон объекта x (в рассчетах не используется, просто для вывода информации в методе __str__)
    def a(self):
        return self.A[-1] if self.A is not None else None

    def activation_derivative(self, Z):
        return self.activation(Z, derivative=True)

    # Метод Forward pass (прогон данных через нейрон)
    def forward(self, Inputs, training=True):
        self.Inputs = Inputs
        self.Z = np.dot(self.Inputs, self.weights) + np.float64(self.bias or 0)  # векторизованное вычисление
        self.A = self.activation(self.Z)
        return self.A
    
    def call(self, Inputs, training=False):
        ''' # Метод call вызывается после выполнения встроенного метода __call__.
        Служит для дополнения/изменения работы встроенного метода __call__ не нарушая необходимый для правильной работы процесс
        вычисления данных при вызове объекта. На вход получает входные данные (массив X). По умолчанию возвращает значение(я) 
        активированного выхода нейрона (массив A).
        '''
        # Любой дополнительный код
        return self.A

    def __call__(self, Inputs, training=False):
        self.forward(Inputs, training)
        return self.call(Inputs, training)

    def __str__(self):
        if self.bias:
            return f'Neuron {self.number}, inputs: {self.inputs}, weights: {self.weights}, bias: {float(self.bias):.4f}, z: {self.z:.4f}, activation: {self.activation.__name__}, a: {self.a:.4f}'
        else:
            return f'Neuron {self.number}, inputs: {self.inputs}, weights: {self.weights}, z: {self.z:.4f}, activation: {self.activation.__name__}, a: {self.a:.4f}'