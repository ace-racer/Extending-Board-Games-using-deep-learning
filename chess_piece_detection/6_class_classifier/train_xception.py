import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import os
import random
from collections import Counter, defaultdict
from itertools import product, combinations
import math
import cv2
from sklearn.model_selection import train_test_split

random.seed(100)

import keras
from keras.layers import Input, Conv2D, Lambda, average, Dense, Flatten,MaxPooling2D, BatchNormalization, Dropout, Activation, Subtract, subtract, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy.random as rng
from keras.applications.xception import Xception


IMAGE_SIZE = (80, 80)
TOTAL_TRAIN_IMAGES = 10000
TOTAL_TEST_IMAGES = 1000

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


# batch size
batch_size = 64

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\\combined_data\\train1',  # this is the target directory
        target_size=IMAGE_SIZE,  
        batch_size=batch_size,
        class_mode='sparse')  

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\\combined_data\\test1',
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='sparse')


# number of training epochs
epochs1 = 500
epochs2 = 200

required_input_shape = (*IMAGE_SIZE, 3)

model_folder_name = "models"
tensorboard_logs_folder_location = "logs"

# checkpoint
if not os.path.exists(model_folder_name):
    os.makedirs(model_folder_name)

# tensorboard logs
if not os.path.exists(tensorboard_logs_folder_location):
    os.makedirs(tensorboard_logs_folder_location)

def get_xception_model():
    base_model = Xception(include_top=False, weights='imagenet', input_shape=required_input_shape)
    # add a global spatial average pooling layer

    print("Shape")
    x = base_model.output
    print(x.shape)

    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a softmax layer
    predictions = Dense(6, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model



xception_model = get_xception_model()


filepath = os.path.join(model_folder_name, "xception.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='max')

tensorboard = TensorBoard(log_dir=tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [checkpoint, earlystop, tensorboard]

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
xception_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# xception_model.fit_generator(
#         train_generator,
#         steps_per_epoch=TOTAL_TRAIN_IMAGES // batch_size,
#         epochs=epochs1,
#         validation_data=validation_generator,
#         validation_steps=TOTAL_TEST_IMAGES // batch_size,
#         callbacks=callbacks_list)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(xception_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit_generator(        train_generator,
#         steps_per_epoch=TOTAL_TRAIN_IMAGES // batch_size,
#         epochs=epochs2,
#         validation_data=validation_generator,
#         validation_steps=TOTAL_TEST_IMAGES // batch_size,
#         callbacks=callbacks_list)
