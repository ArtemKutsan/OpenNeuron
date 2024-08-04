from .units import Neuron, Unit
from .initializers import *
from .activations import *


# Класс для слоя
class Layer:
    count = 0

    def __init__(self, neurons, activation=linear, kernel=GlorotUniform()):
        Layer.count += 1
        self.number = Layer.count  # назначение номера слоя
        self.name = self.__class__.__name__
        self.id = id(self)
        self.output_size = None
        self.Inputs = None  # матрица входов (батча)
        self.kernel = kernel if kernel is not None else GlorotUniform()
        self.weights = None  # матрица весов слоя (веса всех нейронов слоя)
        self.bias = None  # матрица смещений слоя (смещения всех нейронов слоя)
        self.Z = None  # массив значений z всего батча прошедшего через слой (массив векторов значений Z всех нейронов вслое)
        self.activation = activation if activation is not None else linear  # функция активации слоя (всех нейронов в слое)
        self.A = None  # массив значений активаций всего батча прошедшего через слой (массив векторов активаций всех нейронов вслое)
        self.delta = None  # матрица дельт слоя
        self.gradient = {'weights': None, 'bias': None}  # матрица градиентов весов и смещений слоя
        self.neurons = neurons
        self.num_neurons = None

    def initialize(self, input_size):
        self.input_size = input_size
        if isinstance(self.neurons, int):
            # print('Мы в автоматической инициализации нейронов')
            self.output_size = self.neurons
            self.weights = self.kernel(shape=(self.input_size, self.output_size)) 
            self.bias = self.kernel(shape=(1, self.output_size))
            self.neurons = [Unit() for i in range(self.neurons)]
        else:
            # print('Мы в ручной инициализации нейронов')
            self.output_size = len(self.neurons)
            self.weights = np.array([neuron.weights for neuron in self.neurons]).T
            self.bias = np.array([neuron.bias for neuron in self.neurons]).reshape(1, -1)  # проверить .T
        
        self.num_neurons = self.output_size

        ''' # !!! Примечание к матрице весов и смещений слоя.
        Так как Python при работе с изменяемыми объектами (например: list, dict, set и др.) при присваивании одной переменной значения 
        другой переменной просто создает ссылку на значение этого объекта в памяти, то мы можем использовать это. В данном случае 
        в neuron.weights и в neuron.bias не записываются соответствующие значения из матрицы весов и матрицы смещений слоя, 
        а записываются ссылки на эти данные. Таким образом при изменении значений в матрице весов слоя и матрице смещений слоя 
        эти изменения будут и в neuron.weights и neuron.bias и не нужно постоянно вручную перезаписывать эти переменные в каждом нейроне. 
        Это удобно так как для ускорения работы мы работаем с матрицами весов и смещений слоя а не с каждым отдельным нейроном 
        при изменении весов и смещений. При этом нельзя переинициализировывать матрицу весов и смещений заново в коде. Иначе нужно 
        пересоздавать ссылки. '''
        # !!! Создаем/пересоздаем в нейронах ССЫЛКИ на соответствующие веса и смещения матрицы весов и матрицы смещений слоя
        for index, (neuron, weights, bias) in enumerate(zip(self.neurons, self.weights.T, self.bias.T)):
            neuron.weights = weights  # ссылка на веса нейрона в матрице весов слоя
            neuron.bias = bias  # ссылка на смещение нейрона в матрице смещений слоя
            neuron.index = index  # индекс нейрона в слое
            neuron.layer = self  # ссылка на слой в котором находится нейрон
            # Попутно проверяем активации нейрона и если она равна None то устанавливаем ему активацию слоя
            if neuron.activation is None:
                if self.activation == softmax:
                    neuron.activation = linear  # активация нейронов слоя softmax не нужна, то есть A = Z
                else:
                    neuron.activation = self.activation  # если у нейрона не установлена активация то ставим ему активацию слоя 
        return self.output_size

    # Метод Forward pass через слой   
    def forward(self, Inputs, training=True):
        self.Inputs = Inputs
        # (Требуется доработка) Матрица Z вычисляется через матричные вычисления с учетом bias каждого нейрона, матрица A вычисляется БЕЗ учета настройки активации каждого нейрона
        self.Z = np.dot(self.Inputs, self.weights) + np.where(self.bias == None, 0, self.bias).astype(float)
        self.A = self.activation(self.Z)  # активация слоем (одна и та же для всех нейронов одного слоя)
        ''' # Отладка
        if training:
            print(f'Z слоя {self.number} для батча вычисленное матрично:\n{self.Z.T}, {self.Z.dtype}')
            print(f'A слоя {self.number} для батча (одна общая активация слоя для всех нейронов в слое):\n{self.A.T}')
        '''
        ''' # Матрица Z вычисляется с учетом настройки bias каждого нейрона, матрица A вычисляется с учетом настройки активации каждого нейрона
        Z, A = [], []
        for neuron in self.neurons:
            neuron.forward(Inputs, training)
            Z.append(neuron.Z)
            A.append(neuron.A)
        self.Z = np.array(Z).T  # матрица Z значений z всех нейронов в слое (для матричных вычислений)
        self.A = np.array(A).T  # матрица A активаций a всех нейронов в слое (для матричных вычислений)
        # print('Вычисленное в нейронах Z:', self.Z, self.Z.dtype)
        # print('Активация каждого нейрона отдельно A:', self.A)
        '''
        return self.A
       
    # Метод Backward pass через слой   
    def backward(self, delta):
        # Вычисление производной для каждого элемента Z через соответствующий нейрон (сделано для учета функции активации каждого нейрона)
        # derivatives = np.array([[neuron.activation_derivative(z) for z in neuron.Z] for neuron in self.neurons]).T
        
        # Вычисление производной без учета производных активаций каждого нейрона (быстрее но активация всех нейронов в слое одна и та же)
        derivatives = self.activation_derivative(self.Z)
        
        # delta текущего слоя
        self.delta = delta * derivatives

        # Вычисление градиента
        self.gradient['weights'] = np.dot(self.Inputs.T, self.delta)  # градиент весов
        self.gradient['bias'] = np.sum(self.delta, axis=0, keepdims=True)  # градиент смещений

        # delta для следующего (предыдущего) слоя
        # delta = np.dot(self.delta, self.weights.T)
        return np.dot(self.delta, self.weights.T)  # возвращаем delta для следующего (предыдущего) слоя

    def update(self, optimizer):
        optimizer.update(self)

    def activation_derivative(self, Z):
        return self.activation(Z, derivative=True)

    def __str__(self):
        return f'{self.name} {self.number}, neurons: {len(self.neurons)}, inputs: {self.Inputs[-1]}, activation: {self.activation.__name__}, output: {self.A[-1]}'


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

    def backward(self, delta):
        return delta * self.mask / (1 - self.dropout_rate)

    def update(self, optimizer):
        pass  # Dropout не имеет параметров для обновления