# Import necessary components to build LeNet
# Reference: https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Other imports
import numpy as np
import os

# custom imports
import appconfigs
import modelconfigs
import constants
import utils

def train_alexnet_model(model_configs, train_model=True, num_samples=None):

    print("Alexnet model...")

    X_train, y_train = utils.get_required_data_with_labels_for_CNN(appconfigs.location_of_train_data, num_samples, dimensions=(224, 224))

    X_test, y_test = utils.get_required_data_with_labels_for_CNN(appconfigs.location_of_test_data, num_samples, dimensions=(224, 224))


    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=(224, 224, 3),
                       padding='same', kernel_regularizer=l2(0.)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(constants.num_output_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    batch_size = model_configs["batch_size"][0]
    
    # number of training epochs
    nb_epoch = model_configs["epochs"][0]

    if train_model:
        filepath = os.path.join(appconfigs.model_folder_location, model_configs["model_weights_file_name"][0])
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                    save_best_only=True,
                                    mode='max')

        earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10,
                                verbose=1, mode='max')
        
        tensorboard = TensorBoard(log_dir=appconfigs.tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)
        
        callbacks_list = [checkpoint, earlystop, tensorboard]
        
        adam = Adam(lr=model_configs["lr"][0], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        alexnet.compile(loss='sparse_categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

        hist = alexnet.fit(X_train, y_train, shuffle=True, batch_size=batch_size,
                        epochs=nb_epoch, verbose=1,
                        validation_data=(X_test, y_test), callbacks=callbacks_list)

        return hist, alexnet, X_test, y_test
    else:
        adam = Adam(lr=model_configs["lr"][0], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        alexnet.compile(loss='sparse_categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])
        return None, alexnet, X_test, y_test
