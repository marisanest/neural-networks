import numpy as np


class Initializer(object):
    
    def get_fans(shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    
    def normal(shape, scale=0.05):
        return np.random.normal(loc=0.0, scale=scale, size=shape)
    
    @classmethod
    def glorot_normal(cls, shape):
        ''' Reference: Glorot & Bengio, AISTATS 2010
        '''
        fan_in, fan_out = cls.get_fans(shape)
        s = np.sqrt(2. / (fan_in + fan_out))
        return cls.normal(shape, s)
    
    @classmethod
    def he_normal(cls, shape):
        ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
        '''
        fan_in, fan_out = cls.get_fans(shape)
        s = np.sqrt(2. / fan_in)
        return cls.normal(shape, s)