# Keras and TF imports
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
from keras.optimizers import SGD, RMSprop

# Other imports
import numpy as np
import os

# custom imports
import appconfigs
import modelconfigs
import constants

def train_InceptionV3_transfer_learning_model(X_train, y_train, X_test, y_test, model_configs, train_model = True):
    
    print("InceptionV3 model...")
    epochs = model_configs["epochs"][0]
    batch_size = model_configs["batch_size"][0]
    lrs = model_configs["lr"]

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
    
    if train_model:
        print("Training model...")

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in inception_v3_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=RMSprop(lr=lrs[0]), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        # set the list of call backs
        filepath=os.path.join(appconfigs.model_folder_location, "chess_pieces_inceptionv3_p1.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_acc', patience=25, min_delta= 0.0001)
        tensorboard = TensorBoard(log_dir=appconfigs.tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)
        callbacks_list = [checkpoint, early_stopping, tensorboard]

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
        model.compile(optimizer=SGD(lr=lrs[1], momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # checkpoint
        filepath=os.path.join(appconfigs.model_folder_location, "chess_pieces_inceptionv3_p2.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_acc', patience=25, min_delta= 0.001)
        callbacks_list = [checkpoint, early_stopping, tensorboard]

        epochs = model_configs["epochs"][1]
        batch_size = model_configs["batch_size"][1]
        
        ## TODO: replace with fit generator
        history = model.fit(X_train,
                y_train,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=callbacks_list,
                batch_size=batch_size)

        return history, model
    else:
        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=lrs[1], momentum=0.9), metrics=['accuracy'])
        return None, model

def train_custom_cnn_model(X_train, y_train, X_test, y_test, model_configs, train_model = True):
    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid',
                     input_shape=shape_ord))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    model.add(Conv2D(nb_filters * 4, (nb_conv, nb_conv)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()

    # checkpoint
    if not os.path.exists(model_folder_name):
        os.makedirs(model_folder_name)
    
    filepath = os.path.join(model_folder_name, "custom_cnn.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True,
                                 mode='max')

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10,
                              verbose=1, mode='max')
    
    tensorboard = TensorBoard(log_dir=tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)
    
    callbacks_list = [checkpoint, earlystop, tensorboard]
    
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,
                     epochs=nb_epoch, verbose=1,
                     validation_data=(X_test, Y_test), callbacks=callbacks_list)

    # Evaluating the model on the test data
    score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', accuracy)
    return hist, model