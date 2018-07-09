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
    '''         
    Weights will be initilized with ... ?.
    Biases will be initilized with zero.
    '''
    def __init__(self, X_dim, filter_num, filter_dim, stride, padding):
        self.X_dim = X_dim
        self.filter_num = filter_num
        self.filter_dim = filter_dim
        self.stride = stride
        # TODO: check why at first X_dim[2] and then X_dim[3] is used. Add exception for paddings with decimal value 
        self.padding = int(((X_dim[2] - 1) * stride - X_dim[3] + filter_dim) / 2) if padding else 0
        self.W = np.random.randn(filter_num, X_dim[1], filter_dim, filter_dim)
        self.b = np.zeros((filter_num, 1, 1, 1))
        self.params = [self.W, self.b]

    def forward(self, X):
        """
        """
        self.X = X
        self.X_dim = X.shape
        
        # The dimensions of X.
        n, in_C, in_H, in_W = self.X_dim
        
        # The dimensions of output.
        out_C = self.filter_num
        out_H = int((in_H - self.filter_dim + 2 * self.padding) / self.stride) + 1
        out_W = int((in_W - self.filter_dim + 2 * self.padding) / self.stride) + 1
        
        # Initialize output with the correct shapes and zeros.
        out = np.zeros((n, out_C, out_H, out_W))
        
        # Add zero padding to X.
        padded_X = np.pad(X,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant',constant_values = 0)
        
        for i in range(n):                               # Loop over the batch of training samples.
            for c in range(out_C):                       # Loop over channel indices of the output volume.
                for h in range(out_H):                   # Loop over height indices of the output volume.
                    for w in range(out_W):               # Loop over width indices of the output volume.
                        
                        # Corner indices of the window.
                        x_h_start = h * self.stride
                        x_h_end = x_h_start + self.filter_dim
                        x_w_start = w * self.stride
                        x_w_end = x_w_start + self.filter_dim
                    
                        # Select ith training sample and the current window.
                        padded_x_slice = padded_X[i,:,x_h_start:x_h_end,x_w_start:x_w_end]
                        out[i,c,h,w] = np.sum(padded_x_slice * self.W[c,:,:,:]) + self.b[c,:,:,:]
                    
        return out

    def backward(self, dout):
        """Implement the backward propagation for a convolution layer.
    
        Args:
            dout (ndarray): Gradient with respect to the output of the conv layer
        Returns:
            dX (ndarray): Gradient with respect to the input of the conv layer (X)
            dW (ndarray): Gradient with respect to the weights of the conv layer (W)
            db (ndarray): Gradient with respect to the biases of the conv layer (b)
        """
        
        # The dimensions of X.
        n, in_C, in_H, in_W = self.X_dim
        
        # The dimensions of dout.
        n, out_C, out_H, out_W = dout.shape
          
        # Add zero padding to X.
        padded_X =  np.pad(self.X,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant',constant_values = 0)
        
        # Initialize dX, padded_dX dW, db with the correct shapes and zeros.
        dX = np.zeros(self.X.shape)
        padded_dX = np.zeros(padded_X.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)


        # Calculate dX, dW and db
        for i in range(n):                                # Loop over the batch of training samples.
            for c in range(out_C):                        # Loop over the channels of the output volume.
                for h in range(out_H):                    # Loop over height indices of the output volume.
                    for w in range(out_W):                # Loop over width indices of the output volume.
                    
                        # Corner indices of the window.
                        x_h_start = h * self.stride
                        x_h_end = x_h_start + self.filter_dim
                        x_w_start = w * self.stride
                        x_w_end = x_w_start + self.filter_dim
                    
                        # Select ith training sample and the current window.
                        padded_x_slice = padded_X[i,:,x_h_start:x_h_end,x_w_start:x_w_end]
                        
                        # Update gradients.
                        padded_dX[i,:,x_h_start:x_h_end,x_w_start:x_w_end] += dout[i, c, h, w] * self.W[c,:,:,:]
                        
                        dW[c,:,:,:] += dout[i, c, h, w] * padded_x_slice
                
                # Calculate db     
                db[c] = np.sum(dout[:,c,:,:])
       
        # Delete padding 
        dX = padded_dX[:,:,self.padding:-self.padding,self.padding:-self.padding]
    
        return dX, [dW, db]


class Pool():
    ''' Description
    '''
    def __init__(self, X_dim, func, filter_dim, stride):
        self.X_dim = X_dim
        self.func = func
        self.filter_dim = filter_dim
        self.stride = stride
        self.params = []
        
    def forward(self, X):
        """
        """
        self.X = X
        self.X_dim = X.shape
        
        # The dimensions of X.
        n, in_C, in_H, in_W = self.X_dim
        
        # The dimensions of output.
        out_H = int((in_H - self.filter_dim) / self.stride) + 1
        out_W = int((in_W - self.filter_dim) / self.stride) + 1
        out_C = in_C
        
        # Initialize output with the correct shapes and zeros.
        out = np.zeros((n, out_C, out_H, out_W))
        
        for i in range(n):                               # Loop over the batch of training samples.
            for c in range(out_C):                       # Loop over channel indices of the output volume.
                for h in range(out_H):                   # Loop over height indices of the output volume.
                    for w in range(out_W):               # Loop over width indices of the output volume.
                
                        # Corner indices of the window.
                        x_h_start = h * self.stride
                        x_h_end = x_h_start + self.filter_dim
                        x_w_start = w * self.stride
                        x_w_end = x_w_start + self.filter_dim
                    
                        # Select ith training sample and the current window.
                        x_slice = X[i,c,x_h_start:x_h_end,x_w_start:x_w_end]
                        # Calculate output
                        out[i,c,h,w] = self.func(x_slice)
        return out
        
    def backward(self, dout):
        """Implement the backward propagation for a pooling layer.
    
        Args:
            dout (ndarray): Gradient with respect to the output of the pool layer
        Returns:
            dX (ndarray): Gradient with respect to the input of the pool layer (X)
        """
        # The dimensions of dout.
        n, C, out_H, out_W = dout.shape
          
        # Initialize dX with the correct shapes and zeros.
        dX = np.zeros(self.X_dim)
        
        #print(dout[0,0,0,0])
        #print(X[0,0,0:3,0:3])
    
        for i in range(n):                              # Loop over the batch of training samples.
            for c in range(C):                          # Loop over channel indices of the output volume.
                for h in range(out_H):                  # Loop over height indices of the output volume.
                    for w in range(out_W):              # Loop over width indices of the output volume.
                    
                        # Corner indices of the window.
                        x_h_start = h * self.stride
                        x_h_end = x_h_start + self.filter_dim
                        x_w_start = w * self.stride
                        x_w_end = x_w_start + self.filter_dim
                    
                        # Select ith training sample and the current window.
                        x_slice = self.X[i,c,x_h_start:x_h_end,x_w_start:x_w_end]
                        # Calculate mask
                        mask = (x_slice == np.max(x_slice)) 
                        # What is when not only one pxel has the max value? 
                        dX[i,c,x_h_start:x_h_end,x_w_start:x_w_end] = mask * dout[i,c,h,w]
              
        return dX, []


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
        '''
        '''
        self.prob = prob
        self.params = []

    def forward(self, X):
        self.X = X
        
        # Create mask 
        self.mask = np.random.rand(*self.X.shape) < self.prob
        
        # Apply mask
        out = X * self.mask
        
        return out

    def backward(self, dout):
        dX = dout * self.mask * self.prob
        
        return dX, []

