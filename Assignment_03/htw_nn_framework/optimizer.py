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
        return None

    def rmsprop(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with adam
        '''
        return None

    def adam(network, X_train, y_train, loss_function, batch_size=32, epoch=100, learning_rate=0.001, X_test=None, y_test=None, verbose=None):
        '''  Optimize a given network with adam
        '''
        return None

