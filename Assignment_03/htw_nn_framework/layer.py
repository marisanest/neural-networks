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
        self.padding = int(((X_dim[2] - 1) * stride - X_dim[3] + filter_dim) / 2) if padding else 0
        self.W = np.random.randn(filter_num, X_dim[1], filter_dim, filter_dim)
        self.b = np.zeros((filter_num, 1, 1, 1))
        self.params = [self.W, self.b]

    def forward(self, X):
        """
        """
        self.X = X
        self.X_dim = self.X.shape
        
        # The dimensions of X.
        (n, in_C, in_H, in_W) = self.X_dim
        
        # The dimensions of output.
        out_C = self.filter_num
        out_H = int((in_H - self.filter_dim + 2 * self.padding) / self.stride) + 1
        out_W = int((in_W - self.filter_dim + 2 * self.padding) / self.stride) + 1
        
        # Initialize output with the correct shapes and zeros.
        output = np.zeros((n, out_C, out_H, out_W))
        
        # Add zero padding to X.
        padded_X = self.zero_padding(X)
        
        for i in range(n):  # Loop over the batch of training samples.
            
            padded_x = padded_X[i] # Select ith training sample.
          
            for h in range(out_H):  # Loop over height indices of the output volume.
                for w in range(out_W):  # Loop over width indices of the output volume.
                    
                    # Corner indices of the current training sample part.
                    x_h_start = h * self.stride
                    x_h_end = x_h_start + self.filter_dim
                    x_w_start = w * self.stride
                    x_w_end = x_w_start + self.filter_dim
                    
                    # Slice the current training sample part with respect to the corner indices.
                    padded_x_slice = padded_x[:,x_h_start:x_h_end,x_w_start:x_w_end]
                
                    for c in range(out_C): # Loop over channel indices of the output volume.
                        output[i,c,h,w] = np.sum(padded_x_slice * self.W[c,:,:,:] + self.b[c,:,:,:])
                    
        return output

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
        (n, in_C, in_H, in_W) = self.X_dim
        
        # The dimensions of dout.
        (n, out_C, out_H, out_W) = dout.shape
          
        # Initialize dX, dW, db with the correct shapes and zeros.
        dX = np.zeros((n, in_C, in_H, in_W))                           
        dW = np.zeros((out_C, in_C, self.filter_dim, self.filter_dim))
        db = np.zeros((out_C, 1, 1, 1))

        # Add zero padding to X and dX.
        padded_X =  self.zero_padding(self.X)
        padded_dX = self.zero_padding(dX)
        
        for i in range(n):                       # Loop over the batch of training samples.
        
            # Select ith training sample from padded_X and padded_dX.
            padded_x = padded_X[i]
            padded_dx = padded_dX[i]
        
            for h in range(out_H):                   # Loop over height indices of the output volume.
                for w in range(out_W):               # Loop over width indices of the output volume.
                    
                    # Corner indices of the current training sample part.
                    x_h_start = h * self.stride
                    x_h_end = x_h_start + self.filter_dim
                    x_w_start = w * self.stride
                    x_w_end = x_w_start + self.filter_dim
                    
                    # Slice the current training sample part with respect to the corner indices.
                    padded_x_slice = padded_x[:,x_h_start:x_h_end,x_w_start:x_w_end]
                        
                    for c in range(out_C):           # Loop over the channels of the output volume.
                        # Update gradients.
                        padded_dx[:,x_h_start:x_h_end,x_w_start:x_w_end] += self.W[c,:,:,:] * dout[i, c, h, w]
                        dW[c,:,:,:] += padded_x_slice * dout[i, c, h, w]
                        db[c,:,:,:] += dout[i, c, h, w]
                    
            # Set the ith training example's dX to the unpaded padded_dx
            dX[i, :, :, :] = padded_dx[:,self.padding:-self.padding,self.padding:-self.padding]
    
        # Making sure output shape is correct
        assert(dX.shape == (n, in_C, in_H, in_W))
    
        return dX, [dW, db]
    
    def zero_padding(self, X):
        """ Add zero padding to all images in X.
    
        Args:
            X (ndarray): Batch of images.
        Returns:
            Padded images.
        """
        return np.pad(X,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant',constant_values = 0)


class Pool():
    ''' Description
    '''
    def __init__(self, X_dim, func, filter_dim, stride):
        self.X_dim = X_dim
        self.func = func
        self.filter_dim = filter_dim
        self.stride = stride
        #self.W = # TODO: init
        #self.b = # TODO: init
        self.params = [self.W, self.b]
        self.output_dim = self.output_dimension() # The dimensions of the Conv output.
        
    def forward(self, X):
        """
        """
        self.X = X
        self.X_dim = self.X.shape
        
        # The dimensions of X.
        (n, in_H, in_W, in_C) = self.X_dim
        
        # The dimensions of output.
        out_H = int((in_H - self.filter_dim + 2 * self.padding) / self.stride) + 1
        out_W = int((in_W - self.filter_dim + 2 * self.padding) / self.stride) + 1
        out_C = self.filter_num
        
        # Initialize output with the correct shapes and zeros.
        output = np.zeros((n, out_H, out_W, out_C))
        
        for i in range(n):  # Loop over the batch of training samples.
            
            x = X[i] # Select ith training sample.
            
            for h in range(out_H):  # Loop over height indices of the output volume.
                for w in range(out_W):  # Loop over width indices of the output volume.
                
                    # Corner indices of the current training sample part.
                    x_h_start = h * self.stride
                    x_h_end = x_h_start + self.filter_dim
                    x_w_start = w * self.stride
                    x_w_end = x_w_start + self.filter_dim
                    
                    # Slice the current training sample part with respect to the corner indices.
                    x_slice = padded_x[x_h_start:x_h_end,x_w_start:x_w_end,:]

                    for c in range(out_C): # Loop over channel indices of the output volume.
                        output[i,h,w,c] = self.pooling_function(x_slice)
        return output
        
    def backward(self, dout):
        """Implement the backward propagation for a pooling layer.
    
        Args:
            dout (ndarray): Gradient with respect to the output of the pool layer
        Returns:
            dX (ndarray): Gradient with respect to the input of the pool layer (X)
        """
        # The dimensions of X.
        (n, in_H, in_W, in_C) = self.X_dim
        
        # The dimensions of dout.
        (n, out_H, out_W, out_C) = dout.shape
          
        # Initialize dX with the correct shapes and zeros.
        dX = np.zeros(self.X_dim)
    
        for i in range(n):                       # Loop over the batch of training samples.
        
            x = self.X[i] # Select ith training sample.
        
            for h in range(out_H):                   # Loop over height indices of the output volume.
                for w in range(out_W):               # Loop over width indices of the output volume.
                    
                    # Corner indices of the current training sample part.
                    x_h_start = h * self.stride
                    x_h_end = x_h_start + self.filter_dim
                    x_w_start = w * self.stride
                    x_w_end = x_w_start + self.filter_dim
                    
                    for c in range(out_C):           # Loop over channel indices of the output volume.
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                        
                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            x_slice = a_prev[x_h_start:x_h_end,x_w_start:x_w_end,c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dX[i,x_h_start:x_h_end,x_w_start:x_w_end,c] += (mask * x_slice)
                        
                        elif mode == "average":
                        
                            # Get the value a from dA (≈1 line)
                            dx = dX
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (self.filter_dim,self.filter_dim)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dX[i,x_h_start:x_h_end,x_w_start:x_w_end,c] += dx
                        

        # Making sure your output shape is correct
        assert(dX.shape == self.X.shape)
    
        return dX


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


