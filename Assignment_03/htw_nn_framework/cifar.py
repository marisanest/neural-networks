import numpy as np
import os
import pickle
import subprocess
import sys

# used to check whether datasets need to be downloaded
from pathlib import Path

class Cifar():

  def __init__(self):
    self.root_folder = 'cifar-10-batches-py'

  def load_CIFAR_batch(self, filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        images = datadict[b'data']
        labels = datadict[b'labels']
        # Create a numpy array with dimensions (n, channel, height, width)
        images = images.reshape(10000, 3, 32, 32).astype("float64")
        # Squash all values into interval [0,1]
        images = images / 256.
        labels = np.array(labels).astype("uint8")
        return images, labels

  '''Downloads dataset if needed'''
  def load(self):
    if not Path(self.root_folder).is_dir():
      print('No Cifar10 data found.')
      print('Downloading data sets...')
      subprocess.call(["sh","get_datasets.sh"])
    else:
      print('Data already downloaded')

  ''''''
  def get(self):
    # Initialized with the dimensions of the data set
    training_images = np.zeros((50000,3,32,32))
    training_labels = np.zeros((50000)).astype("uint8")
    batch_size = 10000

    # Load and decode each batch of the trainings set
    for index, batch in enumerate(range(1,6)):
        f = os.path.join(self.root_folder, 'data_batch_%d' % batch)
        print(f)

        # Boundary indices of the current batch within the overall trainings set
        b_i_start = index * batch_size
        b_i_end = b_i_start + batch_size

        # Load and decode trainigs data and labels
        training_images[b_i_start:b_i_end], training_labels[b_i_start:b_i_end] = self.load_CIFAR_batch(f)

    # Load and decode test data and labels
    testing_images, testing_labels = self.load_CIFAR_batch(os.path.join(self.root_folder, 'test_batch'))

    return training_images, training_labels, testing_images, testing_labels
