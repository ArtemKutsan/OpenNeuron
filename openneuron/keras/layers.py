import numpy as np
import tensorflow as tf

from .units import Neuron


class CustomDense(tf.keras.layers.Layer):
    count = 0

    def __init__(self, neurons, activation=None, use_bias=True, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        CustomDense.count += 1
        self.number = CustomDense.count
        self.neurons = neurons
        self.units = len(neurons) if isinstance(neurons, list) else neurons
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Создание тензора весов слоя и начальная инициализация
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.units), initializer='he_uniform', trainable=True)

        # Проверка на наличие смещений в нейронах переданных в списке
        use_bias = any(neuron.init_bias is not None for neuron in self.neurons) if isinstance(self.neurons, list) else self.use_bias
        if use_bias:
            # Создание тензора смещений слоя и начальная инициализация
            self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='he_uniform', trainable=True)

        # Создание матрицы весов и смещений из установленных вручную в нейроне весов и смещений
        if isinstance(self.neurons, list):
            Weights = np.array([neuron.init_weights for neuron in self.neurons]).T  # матрица всех весов нейронов слоя
            Biases = np.array([neuron.init_bias if neuron.init_bias is not None else 0 for neuron in self.neurons])  # матрица всех смещений нейронов слоя

            # Переинициализация весов и смещений слоя вручную установленными в нейронах весами и смещениями
            self.set_weights([Weights, Biases]) if use_bias else self.set_weights([Weights])
        else:
            self.neurons = []
            for index in range(self.units):
                neuron = Neuron(weights=self.kernel[:, index])
                if self.use_bias:
                    neuron.init_bias = self.bias[index]
                self.neurons.append(neuron)

        # Прописываем в нейронах родительский слой для считывания весов/смещений слоя и некоторые другие аттрибуты унаследованные от слоя
        for i, neuron in enumerate(self.neurons):
            neuron.index = i
            neuron.number = neuron.index + 1
            neuron.layer = self  # ссылка на родительский слой нейрона
            neuron.activation = self.activation if self.activation is not tf.keras.activations.softmax else None  # активация нейрона = активации слоя

    # Вычисление Z и A всего слоя
    def call(self, Inputs):
        self.Inputs = Inputs
        # Вычисление Z (массив значений z всего слоя)
        self.Z = tf.matmul(self.Inputs, self.kernel) + (self.bias if hasattr(self, 'bias') else 0)
        # Вычисление A путем применение активации ко всему слою (массив значений a всего слоя)
        self.A = self.activation(self.Z) if self.activation is not None else self.Z
        return self.A

    def __str__(self):
        return f'name: {self.name}, neurons: {self.units}, activation: {self.activation.__name__ if self.activation else "None"}'
