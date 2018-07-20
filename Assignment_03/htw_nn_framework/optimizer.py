import numpy as np

class Optimizer():
    '''
    Todo:
        - Class description
        - init method
    '''

    def get_minibatches(X, y, batch_size):
        ''' Decomposes data set into small subsets (batch)
        '''
        m = X.shape[0]
        batches = []
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size, :, :, :]
            y_batch = y[i:i + batch_size, ]
            batches.append((X_batch, y_batch))
        return batches

    def sgd(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, X_test=None, y_test=None, verbose=None):
        ''' Optimize a given network with stochastic gradient descent
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch',i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run vanilla sgd update for all learnable parameters in self.params
                for param, grad in zip(network.params, reversed(grads)):
                    for i in range(len(grad)):
                        param[i] += - learning_rate * grad[i]
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def sgd_momentum(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, mu=0.98, nesterov=None, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with stochastic gradient descent with momentum
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        
        # set initial momentum 
        vs = []
        for i in range(len(network.params)):
            vs.append([])
            for j in range(len(network.params[i])):
                vs[i].append(np.zeros(network.params[i][j].shape))         
                
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch',i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run sgd with momentum update for all learnable parameters in self.params
                for param, grad, v in zip(network.params, reversed(grads), vs):
                    for i in range(len(grad)):
                        v[i] = mu * v[i] - learning_rate * grad[i]
                        param[i] += v[i]
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def rmsprop(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, decay=0.9 ,X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with RMSProp
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        
        # set initial grad cache 
        grads_cache = []
        for i in range(len(network.params)):
            grads_cache.append([])
            for j in range(len(network.params[i])):
                grads_cache[i].append(np.zeros(network.params[i][j].shape))
                
        for i in range(epoch):
            loss = 0
            if verbose:
                print('Epoch',i + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run RMSProp update for all learnable parameters in self.params
                for param, grad, grad_cache in zip(network.params, reversed(grads), grads_cache):
                    for i in range(len(grad)):
                        grad_cache[i] = decay * grad_cache[i] + (1 - decay) * grad[i] * grad[i]
                        param[i] += -learning_rate * grad[i] / (np.sqrt(grad_cache[i]) + 1e-7)
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

    def adam(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, beta1=0.9, beta2=0.999, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with adam
        '''
        minibatches = Optimizer.get_minibatches(X_train, y_train, batch_size)
        
        # set initial m and v
        ms, vs  = [], []
        for i in range(len(network.params)):
            ms.append([])
            vs.append([])
            for j in range(len(network.params[i])):
                ms[i].append(np.zeros(network.params[i][j].shape))
                vs[i].append(np.zeros(network.params[i][j].shape))
                
        for i_e in range(epoch):
            loss = 0
            if verbose:
                print('Epoch',i_e + 1)
            for X_mini, y_mini in minibatches:
                # calculate loss and derivation of the last layer
                loss, dout = loss_function(network.forward(X_mini), y_mini)
                # calculate gradients via backpropagation
                grads = network.backward(dout)
                # run adam update for all learnable parameters in self.params
                for param, grad, m, v in zip(network.params, reversed(grads), ms, vs):
                    for i in range(len(grad)):
                        m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]
                        m_unbias = m[i] / (1 - beta1 ** i_e)
                        v_unbias = v[i] / (1 - beta2 ** i_e)
                        param[i] += -learning_rate * m_unbias / (np.sqrt(v_unbias) + 1e-7)
            if verbose:
                train_acc = np.mean(y_train == network.predict(X_train))
                test_acc = np.mean(y_test == network.predict(X_test))
                print("Loss = {0} :: Training = {1} :: Test = {2}".format(loss, train_acc, test_acc))
        return network

