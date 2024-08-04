import numpy as np


class Initializer:
    def __call__(self, shape):
        raise NotImplementedError('Initializer must implement this method')

class GlorotNormal(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * stddev
    
class GlorotUniform(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    
class HeNormal(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        stddev = np.sqrt(2 / fan_in)
        return np.random.randn(*shape) * stddev


class HeUniform(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)


class LecunNormal(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        stddev = np.sqrt(1 / fan_in)
        return np.random.randn(*shape) * stddev


class LecunUniform(Initializer):
    def __call__(self, shape):
        fan_in, _ = shape
        limit = np.sqrt(3 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)


class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        return np.random.normal(loc=self.mean, scale=self.stddev, size=shape)


class RandomUniform(Initializer):
    def __init__(self, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape):
        return np.random.uniform(self.minval, self.maxval, size=shape)


class TruncatedNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        values = np.random.normal(loc=self.mean, scale=self.stddev, size=shape)
        return np.clip(values, self.mean - 2 * self.stddev, self.mean + 2 * self.stddev)


class Constant(Initializer):
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value)


class Identity(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer can only be used for 2D square matrices.")
        return self.gain * np.eye(shape[0])


class Orthogonal(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.gain * q


class VarianceScaling(Initializer):
    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal'):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def __call__(self, shape):
        if self.mode == 'fan_in':
            fan = shape[0]
        elif self.mode == 'fan_out':
            fan = shape[1]
        elif self.mode == 'fan_avg':
            fan = (shape[0] + shape[1]) / 2
        else:
            raise ValueError("Invalid mode: " + self.mode)

        scale = self.scale / max(1., fan)
        if self.distribution == 'truncated_normal':
            stddev = np.sqrt(scale) / .87962566103423978
            return np.clip(np.random.normal(0, stddev, size=shape), -2*stddev, 2*stddev)
        elif self.distribution == 'normal':
            stddev = np.sqrt(scale)
            return np.random.normal(0, stddev, size=shape)
        elif self.distribution == 'uniform':
            limit = np.sqrt(3.0 * scale)
            return np.random.uniform(-limit, limit, size=shape)
        else:
            raise ValueError("Invalid distribution: " + self.distribution)

