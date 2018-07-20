import numpy as np


class Initializer(object):
    
    def get_fans(shape):
        '''Calculate fan in and fan out.
    
        Args:
            shape: The shape of the volume which should be initialized.
        Returns:
            fan_in (ndarray): The fan in.
            fan_out (ndarray): The fan out.
        '''
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    
    def normal(shape, scale):
        ''' Implements an initialization with normal distributet values.
        
        Args:
            shape: The shape of the volume which should be initialized.
            scale: The scale which the values are scaled with.
        Returns:
            Initialized volume (ndarray)
        '''
        return np.random.randn(*shape) * scale # np.random.normal(scale=scale, size=shape)
    
    @classmethod
    def glorot_normal(cls, shape):
        ''' Implements a Glorot & Bengio initialization.
            Reference: Glorot & Bengio, AISTATS 2010
        
        Args:
            shape: The shape of the volume which should be initialized.
        Returns:
            Initialized volume (ndarray)
        '''
        fan_in, fan_out = cls.get_fans(shape)
        scale = np.sqrt(2. / (fan_in + fan_out))
        return cls.normal(shape, scale)
    
    @classmethod
    def xavier_normal(cls, shape):
        ''' Implements a Xavier initialization.
        
        Args:
            shape: The shape of the volume which should be initialized.
        Returns:
            Initialized volume (ndarray)
        '''
        fan_in, fan_out = cls.get_fans(shape)
        scale = np.sqrt(1. / fan_in)
        return cls.normal(shape, scale)
    
    @classmethod
    def he_normal(cls, shape):
        ''' Implements a modified Xavier initialization (also known as he normal variance scaling initialization).
            Reference: He et al., http://arxiv.org/abs/1502.01852
        
        Args:
            shape: The shape of the volume which should be initialized.
        Returns:
            Initialized volume (ndarray)
        '''
        fan_in, fan_out = cls.get_fans(shape)
        scale = np.sqrt(2. / fan_in)
        return cls.normal(shape, scale)
    
    
    def zero(shape):
        ''' Implements a zero initialization.
        
        Args:
            shape: The shape of the volume which should be initialized.
        Returns:
            Initialized volume (ndarray)
        '''
        return np.zeros(shape)

    def one(shape):
        ''' Implements a initialization with one.
        
        Args:
            shape: The shape of the volume which should be initialized.
        Returns:
            Initialized volume (ndarray)
        '''
        return np.ones(shape)