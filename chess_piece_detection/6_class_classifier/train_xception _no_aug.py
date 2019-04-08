import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

import os
import random
from collections import Counter, defaultdict
from itertools import product, combinations
import math
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
from sklearn.metrics import confusion_matrix


IMAGE_SIZE = (100, 100)
TOTAL_TRAIN_IMAGES = 10000

model_folder_name = "models"
tensorboard_logs_folder_location = "logs"

class PrintConfusionMatrix(keras.callbacks.Callback):
    def __init__(self):
        self.best_f1_score = -1

    def on_epoch_end(self, epoch, logs={}):

        # get the predictions providing the X for the validation data
        y_pred = self.model.predict(self.validation_data[0])
        y_test_pred = [np.argmax(x) for x in y_pred]
        if epoch % 5 == 0:
            print(confusion_matrix(self.validation_data[1], y_test_pred))

        current_f1_score = f1_score(self.validation_data[1], y_test_pred, average='weighted')
        if current_f1_score > self.best_f1_score:
            self.best_f1_score = current_f1_score
            with open(os.path.join(model_folder_name, "f1.txt"), "a") as fp:
                fp.write("Better F1 score {0} obtained during epoch {1} \n".format(current_f1_score, epoch))

            self.model.save_weights(os.path.join(model_folder_name, "best_f1.hdf5"))
        return

# batch size
batch_size = 32

folders_to_create = [model_folder_name, tensorboard_logs_folder_location, "generated"]

for folder_name in folders_to_create:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# number of training epochs
epochs1 = 500
epochs2 = 500

required_input_shape = (*IMAGE_SIZE, 3)

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

def resize_image(image_location):
    image = cv2.imread(image_location)

    if image.shape[0] != IMAGE_SIZE[0] or image.shape[1] != IMAGE_SIZE[1]:
        # print("Resizing the image: {0}".format(image_location))
        resized_image = cv2.resize(image, IMAGE_SIZE, interpolation = cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image

def get_data_from_path(data_path):
    X, y = [], []
    features_with_labels = []
    type_locations = ["b", "n", "k", "p", "q", "r"]
    type_name_to_label = { "p":0, "b":1, "n":2, "r":3, "q": 4, "k":5 }

    for type_name in type_locations:
            piece_type_folder = os.path.join(data_path, type_name)
            for f in (os.listdir(piece_type_folder)):
                if f.endswith(".jpg"):

                    img_file_loc = os.path.join(piece_type_folder, f)
                    resized_image = resize_image(img_file_loc)
                    label = type_name_to_label[type_name]
                    features_with_labels.append({"feature": resized_image, "label": label})

    random.shuffle(features_with_labels)

    X = [x["feature"] for x in features_with_labels]
    y = [x["label"] for x in features_with_labels]

    X = np.array(X)
    X = X.astype('float32')
    X /= 255

    return X, np.array(y)


xception_model = get_xception_model()


filepath = os.path.join(model_folder_name, "xception.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20, verbose=1, mode='max')

tensorboard = TensorBoard(log_dir=tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)

print_confusion_matrix = PrintConfusionMatrix()

callbacks_list = [checkpoint, earlystop, tensorboard, print_confusion_matrix]

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
xception_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

X_train, y_train = get_data_from_path('C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\\six_class_data\\v2\\train')
X_val, y_val = get_data_from_path('C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\\six_class_data\\v2\\test')

hist1 = xception_model.fit(X_train, y_train, shuffle=True, batch_size=batch_size,
                 epochs=epochs1, verbose=1,
                 validation_data=(X_val, y_val), callbacks=callbacks_list)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(xception_model.layers):
#   print(i, layer.name)

# we chose to train the top 3 Xception blocks, i.e. we will freeze
# the first 105 layers and unfreeze the rest:
for layer in xception_model.layers[:106]:
   layer.trainable = False
for layer in xception_model.layers[106:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
xception_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

hist2 = xception_model.fit(X_train, y_train, shuffle=True, batch_size=batch_size,
                 epochs=epochs2, verbose=1,
                 validation_data=(X_val, y_val), callbacks=callbacks_list)

xception_model.save_weights(os.path.join(model_folder_name, "final_xception.hdf5"))