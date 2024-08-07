import numpy as np

from .activations import *
from .losses import *
from .optimizers import *
from .initializers import *
from .utils import format


# Класс для перцептрона
class Perceptron:
    def __init__(self, weights, bias=None):
        self.inputs = None                      # входные значения (подаются в методе __call__)
        self.weights = np.array(weights)        # веса (массив весов)
        self.bias = bias                        # смещение
        self.z = None                           # взвешенная сумма входов + смещение (вычисляется позже в методе __call__)
        self.activation = self.heaviside        # функцию активации перцептрона
        self.a = None                           # результат применения активационной функции (вычисляется позже в методе __call__)

    # Пороговая функция активации
    def heaviside(self, z):
        return 1 if z >= 0.5 else 0

    def __call__(self, inputs):
        self.inputs = np.array(inputs)
        # Рассчитываем взвешенную сумму (z)
        self.z = sum(self.inputs * self.weights) + (self.bias if self.bias is not None else 0)
        # Применяем функцию активации к z
        self.a = self.activation(self.z)
        return self.a

    def __str__(self):
        self(self.inputs) if self.inputs is not None else ...  # рассчитываем актуальные значения z и a
        return f'Perceptron, inputs: {format(self.inputs)}, weights: {format(self.weights)}, bias: {format(self.bias)}, ' \
               f'z: {format(self.z)}, activation: {self.activation.__name__}, a (output): {format(self.a)}'


''' # Старый выриант класса для нейронов
# Класс для нейрона
class Neuron:
    count = 0

    def __init__(self, weights='he', bias='he', input_size=2, activation=linear):
        Neuron.count += 1
        self.number = Neuron.count  # порядковый номер объекта нейрон
        self.name = self.__class__.__name__  # имя класса объекта
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
        # self.Z = np.dot(self.Inputs, self.weights) + (self.bias if self.bias is not None else 0)
        # self.A = self.activation(self.Z)
        return self.A
    
    def call(self, X, training=False):
        """ # Метод call вызывается после выполнения встроенного метода __call__.
        Служит для дополнения/изменения работы встроенного метода __call__ не нарушая необходимый для правильной работы процесс
        вычисления данных при вызове объекта. На вход получает входные данные (массив X). По умолчанию возвращает значение(я) 
        активированного выхода нейрона (массив A).
        """
        # Любой дополнительный код
        return self.A

    def __call__(self, X, training=False):
        self.forward(X, training)
        return self.call(X, training)

    def __str__(self):
        decimals = 4
        return f'{self.name} {self.number}, inputs: {format(self.inputs, decimals=decimals)}, ' \
               f'weights: {format(self.weights.flatten(), decimals=decimals)}, bias: {format(self.bias.item(), decimals=decimals)}, ' \
               f'z: {format(self.z, decimals=decimals)}, activation: {self.activation.__name__}, a: {format(self.a, decimals=decimals)}'
'''
# Класс для нейронов
class Neuron:
    def __init__(self, weights=HeUniform(), bias=HeUniform(), input_size=2, output_size=1, activation=linear, loss=MSE(), optimizer=SGD(momentum=0)):
        self.name = self.__class__.__name__
        self.id = id(self)
        self.Inputs = None
        self.weights = np.array([weights]).T if isinstance(weights, list) else weights(shape=(input_size, output_size))
        self.bias = np.array([[bias]]) if isinstance(bias, (float, int)) else bias(shape=(1, output_size))
        self.Z = None
        self.activation = activation if activation else linear  # функция активации нейрона
        self.A = None
        self.loss_function = loss if loss else MSE()
        self.delta = None
        self.gradient = {'weights': None, 'bias': None}
        self.optimizer = optimizer if optimizer else SGD(momentum=0)

    @property
    def inputs(self):
        return self.Inputs[-1] if self.Inputs is not None else None

    @property
    def z(self):
        return self.Z[-1].item() if self.Z is not None else None
    
    @property
    def a(self):
        return self.A[-1].item() if self.A is not None else None
    
    def activation_derivative(self, Z):
        return self.activation(Z, derivative=True)
    
    def forward(self, X, training=True):
        self.Inputs = X
        self.Z = np.dot(self.Inputs, self.weights) + (self.bias.item() if self.bias.item() is not None else 0)
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, error):
        self.delta = self.loss_function.loss_derivative(error) * self.activation_derivative(self.Z)
        self.gradient['weights'] = np.dot(self.Inputs.T, self.delta)
        self.gradient['bias'] = np.sum(self.delta, axis=0)

    def update(self):
        self.optimizer.update(self)

    def fit(self, X, y, epochs=10, batch_size=1, final_batch_size=None, shuffle=True, validation_data=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.final_batch_size = final_batch_size
        
        if validation_data is not None:
            X_test, y_test = validation_data
            val_data_type = 'Test Data'
        else:
            X_test, y_test = X, y
            val_data_type = 'Train Data'
        
        val_loss = self.loss_function.evaluate_loss(y_test, self.predict(X_test))
        print(f'Training started with Overall Loss {val_loss:.4f} on {val_data_type}')
        
        X_len = X.shape[0]
        for epoch in range(epochs):
            if self.batch_size is not None and self.final_batch_size is not None:
                batch_size = self.batch_size + int((self.final_batch_size - self.batch_size) * (epoch + 1) / epochs)
            
            if shuffle:
                permutation = np.random.permutation(X_len)
                X_shuffled = X[permutation]
                y_shuffled = y[permutation]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, X_len, (batch_size or X_len)):
                num_batches += 1
                X_batch = X_shuffled[i:i+(batch_size or X_len)]
                y_batch = y_shuffled[i:i+(batch_size or X_len)]
                
                predictions_batch = self.forward(X_batch)
                error = self.loss_function.evaluate_error(y_batch, predictions_batch)
                loss = self.loss_function.evaluate_loss(y_batch, predictions_batch)
                epoch_loss += loss
                
                self.backward(error)
                self.update()
            
            loss = epoch_loss / num_batches
            val_loss = self.loss_function.evaluate_loss(y_test, self.predict(X_test))
            
            # Выводим на новой строке только каждую 10-ю эпоху
            end = '\n' if (epochs < 10 or epoch == 0 or (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1) else '\r' 
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f} on {val_data_type}', end=end)
        print(f'Training completed with Overall Loss {val_loss:.4f} on {val_data_type}')

    def predict(self, X):
        return self.forward(X, training=False)
    
    def call(self, X, training=False):
        """ # Метод call вызывается после выполнения встроенного метода __call__.
        Служит для дополнения/изменения работы встроенного метода __call__ не нарушая необходимый для правильной работы процесс
        вычисления данных при вызове объекта. На вход получает входные данные (массив X). По умолчанию возвращает значение(я) 
        активированного выхода нейрона (массив A).
        """
        # Любой дополнительный код
        return self.A

    def __call__(self, X, training=False):
        self.forward(X, training)
        return self.call(X, training)

    def __str__(self):
        decimals, edge_items  = 2, 2
        return f'{self.name}, inputs: {format(self.inputs, decimals=decimals)}, ' \
               f'weights: {format(self.weights.flatten(), decimals=decimals)}, bias: {format(self.bias.item(), decimals=decimals)}, ' \
               f'z: {format(self.z, decimals=decimals)}, activation: {self.activation.__name__}, a: {format(self.a, decimals=decimals)}'


# Класс для нейронов используемых в слое
class Unit:
    count = 0

    def __init__(self, weights='kernel', bias='kernel'):
        Unit.count += 1
        self.number = Unit.count  # порядковый номер объекта класса Unit
        self.name = self.__class__.__name__  # имя класса объекта
        self.id = id(self)  # уникальный id
        self.layer = None  # ссылка на слой в котором находится (устанавливается слоем)
        self.index = None  # индекс в слое (устанавливается слоем)
        self.weights = weights  # веса (могут быть установлены вручную или инициализированы слоем)
        self.bias = bias  # смещение (может быть установлено вручную или инициализировано слоем)
        self.activation = None  # функция активации (устанавливается слоем)

    @property
    # Массив значений inputs (x) батча (X) берется из слоя
    def Inputs(self):
        return self.layer.Inputs
    
    @property
    # Значение inputs последнего прошедшего объекта (в рассчетах не используется, просто для вывода информации в методе __str__)
    def inputs(self):
        return self.Inputs[-1] if self.Inputs is not None else None

    @property
    # Массив значений z батча (z каждого x в батче X) берется из слоя
    def Z(self):
        return self.layer.Z.T[self.index] if self.layer.Z is not None else None
        
    @property
    # Значение z последнего прошедшего объекта (в рассчетах не используется, просто для вывода информации в методе __str__)
    def z(self):
        return self.Z.T[-1].item() if self.Z is not None else None
    
    @property
    # Массив значений активаций батча (a каждого x в батче X)
    def A(self):
        return self.activation(self.Z) if self.Z is not None else None  # собственная активация
        # return self.layer.A.T[self.index] if self.layer.A is not None else None  # активация взятая из активации слоя

    @property
    # Значение активации последнего прошедшего объекта (в рассчетах не используется, просто для вывода информации в методе __str__)
    def a(self):
        return self.A.T[-1].item() if self.A is not None else None
    
    @Inputs.setter
    def Inputs(self):
        raise NotImplementedError('Unsupported method by layer object')

    def activation_derivative(self):
        raise NotImplementedError('Unsupported method by layer object')
    
    def forward(self):
        raise NotImplementedError('Unsupported method by layer object')

    def backward(self):
        raise NotImplementedError('Unsupported method by layer object')

    def update(self):
        raise NotImplementedError('Unsupported method by layer object')

    def fit(self):
        raise NotImplementedError('Unsupported method by layer object')

    def predict(self):
        raise NotImplementedError('Unsupported method by layer object')
    
    def call(self):
        raise NotImplementedError('Unsupported method by layer object')

    def __call__(self):
        raise NotImplementedError('Unsupported method by layer object')
    
    def __str__(self):
        decimals, edge_items = 2, 2
        return f'{self.name} {self.index + 1}/{self.layer.num_neurons}, inputs: {format(self.inputs, decimals, edge_items)}, ' \
               f'weights: {format(self.weights.flatten(), decimals, edge_items)}, bias: {format(self.bias.item(), decimals, edge_items)}, ' \
               f'z: {format(self.z, decimals, edge_items)}, activation: {self.activation.__name__}, a: {format(self.a, decimals, edge_items)}'
    