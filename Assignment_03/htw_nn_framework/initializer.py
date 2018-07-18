import numpy as np


class Initializer(object):
    
    def get_fans(shape):
        '''Calculate fan in and fan out.
    
        Args:
            shape: The shape of the input volume.
        Returns:
            fan_in (ndarray): The fan.
            fan_out (ndarray): The fan out.
        '''
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    
    def normal(shape, scale):
        return np.random.normal(scale=scale, size=shape)
    
    @classmethod
    def glorot_normal(cls, shape):
        ''' Implements a Glorot & Bengio initialisation.
            Reference: Glorot & Bengio, AISTATS 2010
        
        Args:
            shape: The shape of the input volume.
        Returns:
            Initialised ndarray
        '''
        fan_in, fan_out = cls.get_fans(shape)
        scale = np.sqrt(2. / (fan_in + fan_out))
        return cls.normal(shape, scale)
    
    @classmethod
    def he_normal(cls, shape):
        ''' Implements a Xavier initialisation.
            Reference: He et al., http://arxiv.org/abs/1502.01852
        
        Args:
            shape: The shape of the input volume.
        Returns:
            Initialised ndarray
        '''
        fan_in, fan_out = cls.get_fans(shape)
        scale = np.sqrt(2. / fan_in)
        return cls.normal(shape, scale)