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
from keras.optimizers import SGD


IMAGE_SIZE = (100, 100)




model_folder_name = "fine_tuned_models"
tensorboard_logs_folder_location = "fine_tuned_logs"

# checkpoint
if not os.path.exists(model_folder_name):
    os.makedirs(model_folder_name)

# tensorboard logs
if not os.path.exists(tensorboard_logs_folder_location):
    os.makedirs(tensorboard_logs_folder_location)

class PrintConfusionMatrix(keras.callbacks.Callback):
    def __init__(self):
        self.best_f1_score = -1

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        y_test_pred = [np.argmax(x) for x in y_pred]
        if epoch % 5 == 0:
            # get the predictions providing the X for the validation data
            print(confusion_matrix(self.validation_data[1], y_test_pred))

        current_f1_score = f1_score(self.validation_data[1], y_test_pred, average='weighted')
        if current_f1_score > self.best_f1_score:
            self.best_f1_score = current_f1_score
            with open(os.path.join(model_folder_name, "f1.txt"), "a") as fp:
                fp.write("Better F1 score {0} obtained during epoch {1} \n".format(current_f1_score, epoch))

            self.model.save_weights(os.path.join(model_folder_name, "best_f1.hdf5"))
        return

TRAIN_LOC = "C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\sriraj_v6"
VAL_LOC = "C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\\six_class_data\\v1\\test"
EXISTING_MODEL_WEIGHTS = "models/xception.hdf5"

# number of training epochs
epochs = 500

required_input_shape = (*IMAGE_SIZE, 3)



def get_xception_model(weights_location):
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

    # load the model weights
    model.load_weights(weights_location)

    for layer in model.layers[:106]:
        layer.trainable = False

    for layer in model.layers[106:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def resize_image(image_location):
    image = cv2.imread(image_location)

    if image.shape[0] != IMAGE_SIZE[0] or image.shape[1] != IMAGE_SIZE[1]:
        # print("Resizing the image: {0}".format(image_location))
        resized_image = cv2.resize(image, IMAGE_SIZE, interpolation = cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image

type_locations = {"b": ["bb", "wb", "b"], "n": ["bn", "wn", "n"], "k": ["bk", "wk", "k"], "p": ["bp", "wp", "p"], "q": ["bq", "wq", "q"], "r": ["br", "wr", "r"]}
type_name_to_label = { "p":0, "b":1, "n":2, "r":3, "q": 4, "k":5 }

def get_features_labels(data_path):
    X, y = [], []
    features_with_labels = []

    for type_name in type_locations:
        for folder_name in type_locations[type_name]:
            piece_type_folder = os.path.join(data_path, folder_name)

            if os.path.exists(piece_type_folder):
                for f in (os.listdir(piece_type_folder)):
                    if f.endswith(".jpg"):
                        img_file_loc = os.path.join(piece_type_folder, f)
                        print(img_file_loc)
                        processed_image = resize_image(img_file_loc)
                        label = type_name_to_label[type_name]
                        features_with_labels.append({"feature": processed_image, "label": label})

    random.shuffle(features_with_labels)

    X = [x["feature"] for x in features_with_labels]
    y = [x["label"] for x in features_with_labels]

    X = np.array(X)
    X = X.astype('float32')
    X /= 255

    return X, np.array(y)

X_train, y_train = get_features_labels(TRAIN_LOC)
X_test, y_test = get_features_labels(VAL_LOC)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

xception_model = get_xception_model(EXISTING_MODEL_WEIGHTS)


filepath = os.path.join(model_folder_name, "xception.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='max')

tensorboard = TensorBoard(log_dir=tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)

print_confusion_matrix = PrintConfusionMatrix()

callbacks_list = [checkpoint, earlystop, tensorboard, print_confusion_matrix]

# batch size
batch_size = 32

hist = xception_model.fit(X_train, y_train, shuffle=True, batch_size=batch_size,epochs=epochs, verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)
