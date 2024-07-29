from .units import Neuron
from .functions import *
from .activations import *


# Класс для слоя нейронов
class Layer:
    count = 0

    def __init__(self, neurons, activation=None):
        Layer.count += 1
        self.number = Layer.count  # назначение номера слоя
        self.id = id(self)
        self.output_size = None
        self.Inputs = None  # матрица входов (батча)
        self.Weights = None  # матрица весов слоя (веса всех нейронов слоя)
        self.Biases = None  # матрица смещений слоя (смещения всех нейронов слоя)
        self.Z = None  # вектор значений z всего батча прошедшего через слой (массив векторов значений Z всех нейронов вслое)
        self.activation = activation or linear  # функция активации слоя (всех нейронов в слое)
        self.A = None  # вектор значений активаций всего батча прошедшего через слой (массив векторов активаций всех нейронов вслое)
        self.Delta = None  # матрица дельт слоя
        self.Gradient = {'Weights': None, 'Biases': None}  # матрица градиентов весов и смещений слоя
        self.neurons = neurons
        self.num_neurons = None

    def initialize(self, input_size):
        if isinstance(self.neurons, int):
            # print('Мы в автоматической инициализации нейронов')
            self.output_size = self.neurons
            self.Weights, self.Biases = xavier(input_size, self.output_size)
            self.neurons = [Neuron() for i in range(self.neurons)]
        else:
            # print('Мы в ручной инициализации нейронов')
            self.output_size = len(self.neurons)       
            self.Weights = np.array([neuron.weights for neuron in self.neurons]).T
            self.Biases = np.array([neuron.bias for neuron in self.neurons]).reshape(1, -1)
        
        self.num_neurons = self.output_size

        ''' # Важный момент. Примечание к матрице весов и смещений слоя.
        Так как Python при работе с изменяемыми объектами (например: list, dict, set и др.) при присваивании одной переменной значения 
        другой переменной просто создает ссылку на значение этого объекта в памяти, то мы можем использовать это. В данном случае 
        в neuron.weights и в neuron.bias не записываются соответствующие значения из матрицы весов и матрицы смещений слоя, 
        а записываются ссылки на эти данные. Таким образом при изменении значений в матрице весов слоя и матрице смещений слоя 
        эти изменения будут и в neuron.weights и neuron.bias и не нужно постоянно вручную перезаписывать эти переменные в каждом нейроне. 
        Это удобно так как для ускорения работы мы работаем с матрицами весов и смещений слоя а не с каждым отдельным нейроном 
        при изменении весов и смещений. При этом нельзя переинициализировывать матрицу весов и смещений заново в коде. Иначе нужно 
        пересоздавать ссылки. '''
        # !!! Создаем/пересоздаем в нейронах ССЫЛКИ на соответствующие веса и смещения матрицы весов и матрицы смещений слоя
        for neuron, weights, bias in zip(self.neurons, self.Weights.T, self.Biases.T):
            neuron.weights = weights
            neuron.bias = bias
            # Попутно проверяем активации нейрона и если она равна None то устанавливаем им активацию слоя
            if neuron.activation is None:
                if self.activation == softmax:
                    neuron.activation = linear  # для softmax активация нейронов слоя softmax не нужна, то есть A = Z
                else:
                    neuron.activation = self.activation  # если у нейрона не установлена активация то ставим ему активацию слоя 
        return self.output_size

    # Метод Forward pass через слой (реализация со словарями по замерам скорости оказалась быстрее)    
    def forward(self, Inputs, training=True):
        self.Inputs = Inputs
        # Матрица Z вычисляется с учетом настройки bias каждого нейрона, матрица A вычисляется с учетом настройки активации каждого нейрона)
        Z, A = [], []
        for neuron in self.neurons:
            neuron.forward(Inputs, training)
            Z.append(neuron.Z)
            A.append(neuron.A)
        self.Z = np.array(Z).T  # матрица Z значений z всех нейронов в слое (для матричных вычислений)
        self.A = np.array(A).T  # матрица A активаций a всех нейронов в слое (для матричных вычислений)
        return self.A
       
    # Метод Backward pass через слой   
    def backward(self, Delta):
        # Вычисление производной для каждого элемента Z через соответствующий нейрон (сделано для учета функции активации каждого нейрона)
        Derivatives = np.array([[neuron.activation_derivative(z) for z in neuron.Z] for neuron in self.neurons]).T
        # Вычисление производной без учета производных активаций каждого нейрона (быстрее но активация всех нейронов в слое одна и та же)
        # Derivatives = self.activation_derivative(self.Z)  # замерить отличие в скорости
        
        # Delta текущего слоя
        self.Delta = Delta * Derivatives

        # Вычисление градиента
        self.Gradient['Weights'] = np.dot(self.Inputs.T, self.Delta)  # градент весов
        self.Gradient['Biases'] = np.sum(self.Delta, axis=0, keepdims=True)  # градент смещений

        # Delta для следующего слоя
        Next_Delta = np.dot(self.Delta, self.Weights.T)
        return Next_Delta

    def update(self, learning_rate, optimizer=None):
        if optimizer:
            optimizer.update(self, learning_rate)
        else:
            # Обновляем матрицы весов и смещений всего слоя (!!! эти изменения будут также в соответствующих весах и смещегиях нейронов)
            self.Weights -= learning_rate * self.Gradient['Weights']
            # Формируем список индексов элементов Biases которые не равны None
            not_none = np.where(self.Biases != None)
            # Применяем операцию только к элементам массива Biases которые не равны None
            self.Biases[not_none] -= learning_rate * self.Gradient['Biases'][not_none]

    def activation_derivative(self, Z):
        return self.activation(Z, derivative=True)

    def __str__(self):
        return f'Layer {self.number}, neurons: {len(self.neurons)}, activation: {self.activation.__name__}'


# Класс слоя для отключения нейронов 
class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def initialize(self, input_size):
        return input_size  # Dropout не изменяет размер входных данных

    def forward(self, Inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=Inputs.shape)
            return Inputs * self.mask / (1 - self.dropout_rate)
        else:
            return Inputs

    def backward(self, Delta):
        return Delta * self.mask / (1 - self.dropout_rate)

    def update(self, learning_rate, optimizer=None):
        pass  # Dropout не имеет параметров для обновления