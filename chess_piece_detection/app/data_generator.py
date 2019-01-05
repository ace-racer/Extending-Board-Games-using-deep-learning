"""
    Base implementation taken from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import numpy as np
import keras
import os

# Keras/TF imports
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

# custom imports
import constants
import appconfigs


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image_id_split = ID.split("_")
            image_location = os.path.join(appconfigs.base_data_location, image_id_split[0], "_".join(image_id_split[2:]) + ".jpg")
            
            # basic pre-processing of the images
            img = image.load_img(image_location, target_size=self.dim)
            x = image.img_to_array(img)
            x = preprocess_input(x)

            # Store sample
            X[i,] = x

            # Store class
            y[i] = constants.class_names_reverse_mappings[self.labels[ID]]

        return X, y