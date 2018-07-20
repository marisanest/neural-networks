import numpy as np
import os
import pickle
import subprocess
import sys
from os.path import isfile, join, abspath, dirname

# used to check whether datasets need to be downloaded
from pathlib import Path

class Cifar(object):
    ''' Class which loads the CIFAR10 data.
    '''
    # Path of the CIFAR download script
    SCRIPT_PATH =  join(dirname(abspath(__file__)), 'get_datasets.sh')
    
    # Training images dimension
    TRAIN_IMAGES_DIM = (50000,3,32,32)
    # Training labels dimension
    TRAIN_LABELS_DIM = (50000)
    
    # CIFAR batch size
    BATCH_SIZE = 10000
    
    def __init__(self, data_dir):
        # Data directory of the CIFAR batches
        self.data_dir = join(dirname(abspath(__file__)), data_dir)
        # Download data
        self.load()

    def load(self):
        '''Downloads the CIFAR dataset if needed
        '''
        if not Path(self.data_dir).is_dir():
            print('No Cifar10 data found.')
            print('Downloading data sets...')
            # Call a subprocess which downloads the data set
            subprocess.call(["sh", self.SCRIPT_PATH])
        else:
            print('Data already downloaded')

    def get_all_data(self, normalized=False):
        '''Loads all CIFAR data.
        Returns:
            train_images (ndarray): The training images.
            train_labels (ndarray): The corresponding training labels.
            test_images (ndarray): The testing images.
            test_labels (ndarray): The corresponding testing labels.
        '''   
        # Initialized the training images ndarray
        train_images = np.zeros(self.TRAIN_IMAGES_DIM)
        # Initialized the training labels ndarray
        train_labels = np.zeros(self.TRAIN_LABELS_DIM).astype("uint8")

        # Load each batch of the trainings set
        for i, batch in enumerate(range(1,6)):
            filename = os.path.join(self.data_dir, 'data_batch_%d' % batch)
            print('Loading ', filename)
            # Boundary indices of the current batch within the overall trainings set
            start_i = i * self.BATCH_SIZE
            end_i = start_i + self.BATCH_SIZE

            # Load CIFAR batch
            train_images[start_i:end_i], train_labels[start_i:end_i] = self.get_batch(filename, normalized)

        # Load test images and labels
        test_images, test_labels = self.get_batch(os.path.join(self.data_dir, 'test_batch'))

        return train_images, train_labels, test_images, test_labels
    
    def get_batch(self, filename, normalized=False):
        '''Loads CIFAR batch.
        
         Args:
            filename: The filename of the CIFAR batch with.
        Returns:
            images (ndarray): The images.
            labels (ndarray): The corresponding labels.
        '''
        with open(filename, 'rb') as f:
            # Load data from file
            datadict = pickle.load(f, encoding='bytes')
            
            # Extract images
            images = datadict[b'data']
            # Reshapes images to dimension (n, channel, height, width)
            images = images.reshape(10000, 3, 32, 32)
            # Convert image values to floats
            images = images.astype("float64")
            # Normalize data if wanted
            if normalized:
                # Squash all values into interval [0,1]
                images = images / 256.
            
            # Extract labels
            labels = datadict[b'labels']
            # Convert labels to integers
            labels = np.array(labels).astype("uint8")
            
            return images, labels


