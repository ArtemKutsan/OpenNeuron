import numpy as np

from .units import Neuron
from .layers import Layer
from .activations import *
from .losses import *
from .optimizers import *


# Класс для нейронной сети
class NeuralNetwork:
    Neuron.count = 0
    Layer.count = 0
    def __init__(self, layers, inputs_dim, loss='mse', optimizer=SGD(momentum=0)):
        Neuron.count = 0
        Layer.count = 0
        self.type = None
        self.layers = layers
        self.last_layer = self.layers[-1]
        self.inputs_dim = inputs_dim
        self.loss_function = self.init_loss_function(loss)
        self.optimizer = optimizer or SGD(momentum=0)
        self.batch_size = None
        self.final_batch_size = None
        self.epochs = None
        self.initialize(inputs_dim)
    
    def init_loss_function(self, loss_name):
        if loss_name == 'mae':
            return MAE()
        elif loss_name == 'mse':
            return MSE()
        elif loss_name == 'categorical_crossentropy':
            return CE()
        else:
            raise ValueError('Unsupported loss function')
        
    # Метод для инициализации сети (слоев/нейронов)
    def initialize(self, inputs_dim):
        input_size = inputs_dim
        for layer in self.layers:
            # Инициализируем слой исходя из размера входных данных и получаем размерность для следующего слоя исходя из размера текущего слоя
            output_size = layer.initialize(input_size)
            input_size = output_size
        self.type = '"Classification"' if self.last_layer.activation == sigmoid or self.last_layer.activation == softmax else '"Regression"'
    
    # Метод Forward pass через всю сеть  
    def forward(self, X, training=True):
        Inputs = X  # входные данные (могут быть одним объекомт выборки (x), несколькими объектами выборки (X_batch), частный случай - это вся выборка целиком (X))
        # "Прогоняем" данные через все слои
        for layer in self.layers:
            A = layer.forward(Inputs, training)  # подаем входные данные в слой и получаем активацию (выход) слоя
            Inputs = A  # активация (выход слоя) это входные данные для следующего слоя
        # Активация всего последнего/выходного слоя (нужна для softmax в задачах классификации)
        self.last_layer.A = self.last_layer.activation(self.last_layer.Z)
        ''' # Отладочный вывод
        if training:
            print(f'Z нейронов выходного слоя в методе forward всей сети:\n{self.last_layer.Z}')
            print(f'Активация (A) выходного слоя в методе forward всей сети:\n{self.last_layer.A}')
        '''
        return self.last_layer.A  # выход сети (активация последнего/выходного слоя/нейрона)
    
    # Метод Backward pass (обратное распрстранение ошибки) через всю сеть
    def backward(self, error):
        delta = self.loss_function.loss_derivative(error)  # delta выходного слоя/нейрона это производная функции потерь
        # Рассчитываем delta для всех остальных слоев
        for layer in reversed(self.layers):
            delta = layer.backward(delta)  #  передаем delta в слой и получаем новую delta для предыдущего слоя

    def update(self):
        for layer in reversed(self.layers):
            layer.update(self.optimizer)
            
    def fit(self, X, y, epochs, batch_size=1, final_batch_size=None, shuffle=True, validation_data=None): 
        self.epochs = epochs
        self.batch_size = batch_size
        self.final_batch_size = final_batch_size

        # Определяем валидационную выборку        
        if validation_data is not None:
            X_test, y_test = validation_data
            val_data_type = 'Test Data'
        else:
            X_test, y_test = X, y
            val_data_type = 'Train Data'
        
        # Ошибка (потеря) по всей выборке на старте обучения
        val_loss = self.loss_function.evaluate_loss(y_test, self.predict(X_test))
        print(f'Training started with Overall Loss {val_loss:.4f} on {val_data_type}')
        
        self.X_len = X.shape[0]
        for epoch in range(epochs):
            # Расчет текущего размера батча, если batch_size не None (для Stochastic gradient Descent, Mini-batch gradient Descent или Batch gradient Descent)
            if self.batch_size is not None and self.final_batch_size is not None:
                batch_size = self.batch_size + int((self.final_batch_size - self.batch_size) * (epoch + 1) / epochs)
            
            # Перемешиваем данные
            if shuffle:
                permutation = np.random.permutation(self.X_len)
                X_shuffled = X[permutation]
                y_shuffled = y[permutation]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_loss = 0  # ошибка (потеря) по батчам за эпоху
            num_batches = 0
            
            for i in range(0, self.X_len, batch_size or self.X_len):
                num_batches += 1
                # Разделение выдорки на батчи
                X_batch = X_shuffled[i:i+(batch_size or self.X_len)]
                y_batch = y_shuffled[i:i+(batch_size or self.X_len)]
                
                # Forward pass (прямой проход по сети)
                predictions_batch = self.forward(X_batch)
                
                # Ошибка выходного слоя/нейрона
                error = self.loss_function.evaluate_error(y_batch, predictions_batch)

                # Calculate batch loss
                loss = self.loss_function.evaluate_loss(y_batch, predictions_batch)
                epoch_loss += loss
                
                # Backward pass (обратное распространение ошибки)
                self.backward(error)
                
                # Update weights and bias
                self.update()
            
            # Средняя ошибка (потеря) по батчам за эпоху
            loss = epoch_loss / num_batches
            # Ошибка (потеря) по всей выборке в конце каждой эпохи обучения (по тестовой или тренировочной выбоке)
            val_loss = self.loss_function.evaluate_loss(y_test, self.predict(X_test))
            # Выводим на новой строке только каждую 10-ю эпоху
            end = '\n' if (epochs < 10 or epoch == 0 or (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1) else '\r' 
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f} on {val_data_type}', end=end)
            # sys.stdout.flush()  # принудительно сбрасываем буфер вывода
        print(f'Training completed with Overall Loss {val_loss:.4f}')

    def predict(self, X):
        return self.forward(X, training=False)
    
    def summary(self):
        print(self)
        for layer in self.layers:
            if isinstance(layer, Layer): 
                print(layer)
                
                if layer.num_neurons > 4:
                    print(layer.neurons[0])
                    print(layer.neurons[1])
                    print('...')
                    print(layer.neurons[-2])
                    print(layer.neurons[-1])
                else:
                    for neuron in layer.neurons:
                        print(neuron)

    def __str__(self):
        batch_size_str = str(self.batch_size) or str(self.X_len)
        batch_size_str += ('-' + str(self.final_batch_size)) if self.final_batch_size is not None else ''
        return f'Network type: {self.type}, layers: {self.last_layer.number}, neurons: {self.last_layer.neurons[-1].number}, optimizer: {self.optimizer}, epochs: {self.epochs}, batch size: {batch_size_str}, loss function: {self.loss_function}'