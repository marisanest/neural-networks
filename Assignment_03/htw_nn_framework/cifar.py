import numpy as np
import os
import pickle
import subprocess
import sys
from os.path import isfile, join, abspath, dirname

# used to check whether datasets need to be downloaded
from pathlib import Path

class Cifar():
    
    # Root directory of the CIFAR batches
    ROOT_DIR = join(dirname(abspath(__file__)), 'cifar-10-batches-py')
    
    # Path of the CIFAR download script
    SCRIPT_PATH =  join(dirname(abspath(__file__)), 'get_datasets.sh')
    
    # Training images dimension
    TRAIN_IMAGES_DIM = (50000,3,32,32)
    # Training labels dimension
    TRAIN_LABELS_DIM = (50000)
    
    # CIFAR batch size
    BATCH_SIZE = 10000

    def load_CIFAR_batch(filename):
        """Loads the data from a CIFAR batch.
        
         Args:
            filename: The filename of the CIFAR batch with.
        Returns:
            images (ndarray): The images.
            labels (ndarray): The corresponding labels.
        """
        with open(filename, 'rb') as f:
            # Load data from file
            datadict = pickle.load(f, encoding='bytes')
            
            # Extract images
            images = datadict[b'data']
            # Reshapes images to dimension (n, channel, height, width)
            images = images.reshape(10000, 3, 32, 32)
            # Convert image values to floats
            images = images.astype("float64")
            # Squash all values into interval [0,1]
            images = images / 256.
            
            # Extract labels
            labels = datadict[b'labels']
            # Convert labels to integers
            labels = np.array(labels).astype("uint8")
            
            return images, labels

    @classmethod
    def load(cls):
        """Downloads the CIFAR dataset if needed
        """
        if not Path(cls.ROOT_DIR).is_dir():
            print('No Cifar10 data found.')
            print('Downloading data sets...')
            # Call a subprocess which downloads the data set
            subprocess.call(["sh", cls.SCRIPT_PATH])
        else:
            print('Data already downloaded')

    @classmethod
    def get(cls):
        """Loads the whole CIFAR batch.
        Returns:
            tr_images (ndarray): The training images.
            tr_labels (ndarray): The corresponding training labels.
            te_images (ndarray): The testing images.
            te_labels (ndarray): The corresponding testing labels.
        """
        if not Path(cls.ROOT_DIR).is_dir():
            print('No Cifar10 data found. Please load data first!')
            return
        
        # Initialized the training images ndarray
        tr_images = np.zeros(cls.TRAIN_IMAGES_DIM)
        # Initialized the training labels ndarray
        tr_labels = np.zeros(cls.TRAIN_LABELS_DIM).astype("uint8")

        # Load each batch of the trainings set
        for i, batch in enumerate(range(1,6)):
            f = os.path.join(cls.ROOT_DIR, 'data_batch_%d' % batch)
            print(f)
            # Boundary indices of the current batch within the overall trainings set
            start_i = i * cls.BATCH_SIZE
            end_i = start_i + cls.BATCH_SIZE

            # Load CIFAR batch
            tr_images[start_i:end_i], tr_labels[start_i:end_i] = cls.load_CIFAR_batch(f)

        # Load test images and labels
        te_images, te_labels = cls.load_CIFAR_batch(os.path.join(cls.ROOT_DIR, 'test_batch'))

        return tr_images, tr_labels, te_images, te_labels
