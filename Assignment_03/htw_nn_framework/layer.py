import numpy as np

class Flatten():
    ''' Flatten layer used to reshape inputs into vector representation

    Layer should be used in the forward pass before a dense layer to
    transform a given tensor into a vector.
    '''
    def __init__(self):
        self.params = []

    def forward(self, X):
        ''' Reshapes a n-dim representation into a vector
            by preserving the number of input rows.

        Examples:
            [10000,[1,28,28]] -> [10000,784]
        '''
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.reshape(-1).reshape(self.out_shape)
        return out

    def backward(self, dout):
        ''' Restore dimensions before flattening operation
        '''
        out = dout.reshape(self.X_shape)
        return out, []

class FullyConnected():
    ''' Fully connected layer implemtenting linear function hypothesis
        in the forward pass and its derivation in the backward pass.
    '''
    def __init__(self, in_size, out_size):
        ''' Initilize all learning parameters in the layer

        Weights will be initilized with modified Xavier initialization.
        Biases will be initilized with zero.
        '''
        self.W = np.random.randn(in_size, out_size) * np.sqrt(2. / in_size)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = np.add(np.dot(self.X, self.W), self.b)
        return out

    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        dW = np.dot(self.X.T, dout)
        db = np.sum(dout, axis=0)
        return dX, [dW, db]

class Conv():
    ''' Description
    '''
    def __init__(self, X_dim, filter_num, filter_dim, stride, padding):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None


class Pool():
    ''' Description
    '''
    def __init__(self, X_dim, func, filter_dim, stride):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None

class Batchnorm():
    ''' Description
    '''
    def __init__(self, X_dim):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None


class Dropout():
    ''' Description
    '''
    def __init__(self, prob=0.5):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None


