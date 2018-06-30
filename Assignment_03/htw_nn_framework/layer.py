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
        self.X_dim = X_dim
        self.filter_num = filter_num
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.W = # TODO: init
        self.b = # TODO: init
        self.params = [self.W, self.b]
        self.output_dim = self.output_dimension() # The dimensions of the Conv output.

    def forward(self, X):
        """
        """
        # Initialize the output volume with zeros.
        output = np.zeros(self.output_dim)
        
        # Calculate image with padding and new image dimension.
        image, self.image_dim = self.zero_padding(image)
        
        for x_i, x in tqdm(enumerate(X)):  # loop over image samples.
            for h in range(self.output_dim[1]):  # loop over height indices of the output volume.
                for w in range(self.output_dim[2]):  # loop over width indices of the output volume.
                
                    # Indices if the current image part that sould be used for computation.
                    x_H_i_start = h * self.stride
                    x_H_i_end = image_H_i_start + self.filter_dim
                    x_W_i_start = w * self.stride
                    x_W_i_end = image_W_i_start + self.filter_dim
                
                    for f_i, f in enumerate(self.W):
                        output[x_i,h,w,f_i] = np.sum(x[x_H_i_start:x_H_i_end,x_W_i_start:x_W_i_end] * f)
                    
        return output

    def backward(self, dout):
        return None


class Pool():
    ''' Description
    '''
    def __init__(self, X_dim, func, filter_dim, stride):
        self.X_dim = X_dim
        self.func = func
        self.filter_dim = filter_dim
        self.stride = stride
        self.W = # TODO: init
        self.b = # TODO: init
        self.params = [self.W, self.b]
        self.output_dim = self.output_dimension() # The dimensions of the Conv output.
        
    def forward(self, X):
        # Initialize the output volume with zeros.
        output = np.zeros(self.output_dim)
        
        for x_i, x in tqdm(enumerate(X)):  # loop over image samples.
            for h in range(self.output_dim[1]):  # loop over height indices of the output volume.
                for w in range(self.output_dim[2]):  # loop over width indices of the output volume.
                
                    # Indices if the current image part that sould be used for computation.
                    x_H_i_start = h * self.stride
                    x_H_i_end = image_H_i_start + self.filter_dim
                    x_W_i_start = w * self.stride
                    x_W_i_end = image_W_i_start + self.filter_dim
                    
                    for f_i, f in enumerate(self.W):
                        output[x_i,h,w,f_i] = self.pooling_function(x[x_H_i_start:x_H_i_end,x_W_i_start:x_W_i_end,:])
                    
        return output

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


