import numpy as np

class ReLU():
    ''' Implements activation function rectified linear unit (ReLU)

    ReLU activation function is defined as the positive part of
    its argument. Todo: insert arxiv paper reference
    '''
    def __init__(self):
        self.params = []

    def forward(self, X, **kwargs):
        ''' In the forward pass return the identity for x < 0

        Safe input for backprop and forward all values that are above 0.
        '''
        self.X = X
        return np.maximum(X, 0)

    def backward(self, dout):
        ''' Derivative of ReLU

        Retruns:
            dX: for all x \elem X <= 0 in forward pass
                return 0 else x
            []: no gradients on ReLU operation
        '''
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []

class LeakyReLU():
    ''' Implements activation function Leaky ReLU.
    '''
    def __init__(self):
        self.params = []

    def forward(self, X, **kwargs):
        ''' Implement the forward pass for the Leaky ReLU activation function.

        Args:
            X (ndarray): Input of the activation function.
        Returns:
            out (ndarray): Output of the activation function.
        '''
        self.X = X
        return np.maximum(.01 * X, X)

    def backward(self, dout):
        ''' Implement the backward pass for the Leaky ReLU activation function.

        Retruns:
            dX: The gradients with respect to the input of the activation function (X).
            []: No gradients for the weights and biases on Leaky ReLU operation.
        '''
        dX = dout.copy()
        dX[self.X <= 0] = .01 * dout
        return dX, []

class sigmoid():
    ''' Implements activation function sigmoid.
    '''
    def __init__(self):
        self.params = []

    def forward(self, X, **kwargs):
        ''' Implement the forward pass for the sigmoid activation function.

        Args:
            X (ndarray): Input of the activation function.
        Returns:
            out (ndarray): Output of the activation function.
        '''
        self.X = X
        return 1. / (1 + np.exp(-self.X))

    def backward(self, dout):
        ''' Implement the backward pass for the sigmoid activation function.

        Retruns:
            dX: The gradients with respect to the input of the activation function (X).
            []: No gradients for the weights and biases on Leaky ReLU operation.
        '''
        # Splitting function to get a computational graph
        # f = 1 / g 
        # g = 1 + h
        # h = e^-x
        
        # Calculate derevatives with partial derivation and the chain rule
        # df/df = 1 bzw. dout
        # df/dg = df/df * - 1/g^2
        # df/dh = df/dg * 1
        # df/dx = df/dh * e^-x 
        #       = - 1/g^2 * e^-x  
        #       = - 1/(1 + e^-x)^2 * e^-x
        
        dX = dout * -(1. / np.square(1 + np.exp(-self.X))) * np.exp(-self.X)
        return dX, []

class tanh():
    ''' Implements activation function tanh.
    '''
    def __init__(self):
        self.params = []

    def forward(self, X, **kwargs):
        ''' Implement the forward pass for the tanh activation function.

        Args:
            X (ndarray): Input of the activation function.
        Returns:
            out (ndarray): Output of the activation function.
        '''
        self.X = X
        return np.tanh(self.X)

    def backward(self, dout):
        ''' Implement the backward pass for the tanh activation function.

        Retruns:
            dX: The gradients with respect to the input of the activation function (X).
            []: No gradients for the weights and biases on Leaky ReLU operation.
        '''
        # Splitting function to get a computational graph
        # f = 1 - g 
        # g = 2 / h
        # h = 1 + i
        # i = e^j
        # j = 2x
        
        # Calculate derevatives with partial derivation and the chain rule
        # df/df = 1 bzw. dout
        # df/dg = df/df * -1
        # df/dh = df/dg * - 2/h^2
        # df/di = df/dh * 1
        # df/dj = df/di * e^j
        # df/dx = df/dj * 2
        #       = dout * -1 * - 2/h^2 * 1 * e^j * 2
        #       = (dout * 4 * e^2x) / (1 + e^2x)^2
        
        dX = (dout * 4 * np.exp(2 * self.X)) / np.square(1 + np.exp(2 * self.X))
        return dX, []

