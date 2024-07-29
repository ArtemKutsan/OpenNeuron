import numpy as np

from .units import Neuron
from .layers import Layer
from .functions import *


# Класс для нейронной сети
class NeuralNetwork:
    Neuron.count = 0
    Layer.count = 0
    def __init__(self, layers, inputs_dim, loss='mse', optimizer=None):
        Neuron.count = 0
        Layer.count = 0
        self.type = None
        self.layers = layers
        self.last_layer = self.layers[-1]
        self.inputs_dim = inputs_dim
        self.loss_function = self.init_loss_function(loss)
        self.optimizer = optimizer
        self.batch_size = None
        self.final_batch_size = None
        self.epochs = None
        self.initialize(inputs_dim)
    
    def init_loss_function(self, loss_name):
        if loss_name == 'mae':
            self.type = '"Regression"'
            return MAE()
        elif loss_name == 'mse':
            self.type = '"Regression"'
            return MSE()
        elif loss_name == 'categorical_crossentropy':
            self.type = '"Classification"'
            return CE()
        else:
            self.type = 'Unknown'
            raise ValueError('Unsupported loss function')
        
    # Метод для инициализации сети (слоев/нейронов)
    def initialize(self, inputs_dim):
        input_size = inputs_dim
        for layer in self.layers:
            # Инициализируем слой исходя из размера входных данных и получаем размерность для следующего слоя исходя из размера текущего слоя
            output_size = layer.initialize(input_size)
            input_size = output_size
    
    # Метод Forward pass через всю сеть  
    def forward(self, X, training=True):
        Inputs = X
        for layer in self.layers:
            A = layer.forward(Inputs, training)
            Inputs = A
        # Активация всего слоя (нужна для softmax в задачах классификации)
        self.last_layer.A = self.last_layer.activation(self.last_layer.Z)
        # if training:
            # print(f'Z нейронов выходного слоя в методе forward всей сети:\n{self.last_layer.Z}')
            # print(f'Активация (A) выходного слоя в методе forward всей сети:\n{self.last_layer.A}')
        return self.last_layer.A  # выход сети (активация последнего (выходного) слоя/нейрона)
    
    # Метод Fackward pass через всю сеть  
    def backward(self, Delta):
        # Рассчитываем Delta для всех слоев
        for layer in reversed(self.layers):
            # Передаем Delta в слой и получаем новую Delta для предыдущего слоя
            Next_Delta = layer.backward(Delta)
            Delta = Next_Delta

    def update(self):
        for layer in reversed(self.layers):
            layer.update(self.learning_rate, self.optimizer)
            
    def fit(self, X, y, learning_rate, epochs, batch_size=1, final_batch_size=None, shuffle=True, validation_data=(None, None)): 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.final_batch_size = final_batch_size
        self.X_test, self.y_test = validation_data
        
        # Ошибка предсказаний по всей выборке на старте обучения
        if self.X_test is not None and self.y_test is not None:
            val_loss = self.loss_function.evaluate_loss(self.y_test, self.predict(self.X_test))
        else:
            val_loss = self.loss_function.evaluate_loss(y, self.predict(X))
        print(f'Training started with Overall Loss {val_loss:.4f}')
        
        self.X_len = X.shape[0]
        for epoch in range(epochs):
            # Расчет текущего размера батча, если batch_size не None (для Stochastic Gradient Descent, Mini-batch Gradient Descent или Batch Gradient Descent)
            if self.batch_size is not None and self.final_batch_size is not None:
                batch_size = self.batch_size + int((self.final_batch_size - self.batch_size) * (epoch + 1) / epochs)
            
            # Shuffle data
            if shuffle:
                permutation = np.random.permutation(self.X_len)
                X_shuffled = X[permutation]
                y_shuffled = y[permutation]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, self.X_len, batch_size or self.X_len):
                num_batches += 1
                # Get mini-batch
                X_batch = X_shuffled[i:i+(batch_size or self.X_len)]
                y_batch = y_shuffled[i:i+(batch_size or self.X_len)]
                
                # Forward pass
                predictions_batch = self.forward(X_batch)
                
                # Calculate batch loss
                loss = self.loss_function.evaluate_loss(y_batch, predictions_batch)
                epoch_loss += loss
                
                # Delta выходного слоя для начала процесса обратного распространения ошибки
                Delta = self.loss_function.evaluate_delta(y_batch, predictions_batch)
                
                # Backward pass
                self.backward(Delta)
                
                # Update weights and biases
                self.update()
            
            # Средняя ошибка по батчам за эпоху
            loss = epoch_loss / num_batches
            
            # Ошибка предсказаний по всей обучающей выборке за эпоху
            if self.X_test is not None and self.y_test is not None:
                val_loss = self.loss_function.evaluate_loss(self.y_test, self.predict(self.X_test))
            else:
                val_loss = self.loss_function.evaluate_loss(y, self.predict(X))
            
            if epochs < 10 or epoch % (epochs // 10) == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}', end='\r')
            # sys.stdout.flush()  # принудительно сбрасываем буфер вывода
        print(f'Training completed with Overall Loss {val_loss:.4f}')

    def predict(self, X):
        return self.forward(X, training=False)
    
    def __str__(self):
        batch_size_str = str(self.batch_size) or str(self.X_len)
        batch_size_str += ("-" + str(self.final_batch_size)) if self.final_batch_size is not None else ""
        return f'Network type: {self.type}, layers: {self.last_layer.number}, neurons: {self.last_layer.neurons[-1].number}, optimizer: {self.optimizer}, epochs: {self.epochs}, batch size: {batch_size_str}, loss function: {self.loss_function}'