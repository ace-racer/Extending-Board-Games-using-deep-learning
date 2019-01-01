# Keras and TF imports
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
from keras.optimizers import SGD

# Other imports
import numpy as np
import os

# custom imports
import appconfigs
import modelconfigs
import constants

def train_InceptionV3_transfer_learning_model(X_train, y_train, X_test, y_test, model_configs):
    # create the base pre-trained model
    inception_v3_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = inception_v3_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(constants.num_output_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inception_v3_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in inception_v3_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # set the list of call backs
    filepath=os.path.join(appconfigs.model_folder_location, "chess_pieces_inceptionv3_p1.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, min_delta= 0.0001)
    tensorboard = TensorBoard(log_dir=appconfigs.tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, early_stopping, tensorboard]

    epochs = model_configs["epochs"][0]
    batch_size = model_configs["batch_size"][0]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # TODO: replace with fit generator
    _ = model.fit(X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks_list,
            batch_size=batch_size)

    ## Fine tune some inception layers
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # checkpoint
    filepath=os.path.join(appconfigs.model_folder_location, "chess_pieces_inceptionv3_p2.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=25, min_delta= 0.01)
    callbacks_list = [checkpoint, early_stopping, tensorboard]

    epochs = model_configs["epochs"][1]
    batch_size = model_configs["batch_size"][1]

    history = model.fit(X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks_list,
            batch_size=batch_size)

    return history, model

