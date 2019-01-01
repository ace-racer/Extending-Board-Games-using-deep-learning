import cv2
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras import backend as K
from keras import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import set_random_seed
set_random_seed(42)
# set the location of the training and test images (change as required)
location_of_train_data = "C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\Chess ID Public Data\\output_train"
location_of_test_data = "C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\Chess ID Public Data\\output_test"
model_folder_name = "C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\trained_models\\31_12"
tensorboard_logs_folder_location = "C:\\Users\issuser\\Desktop\\ExtendingBoardGamesOnline\\tb_logs\\31_12"

class_names = ["white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook",
               "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook", "empty"]
class_names_reverse_mappings = {"white_bishop": 0, "white_king": 1, "white_knight": 2, "white_pawn": 3, "white_queen": 4, "white_rook": 5,
                                "black_bishop": 6, "black_king": 7, "black_knight": 8, "black_pawn": 9, "black_queen": 10, "black_rook": 11, "empty": 12}
class_names_folder_mappings = {"white_bishop": ["wb"], "white_king": ["wk"], "white_knight": ["wn"], "white_pawn": ["wp"], "white_queen": ["wq"], "white_rook": [
    "wr"], "black_bishop": ["bb"], "black_king": ["bk"], "black_knight": ["bn"], "black_pawn": ["bp"], "black_queen": ["bq"], "black_rook": ["br"], "empty": ["empty"]}






batch_size = 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# number of training epochs
nb_epoch = 150

# convolution kernel size
nb_conv = 3

# shape of each image
shape_ord = (200, 200, 3)

# number of output classes
nb_classes = 13

def train_custom_cnn_model(X_train, Y_train, X_test, Y_test):
    """"""
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

hist, model = train_custom_cnn_model(X_train, y_train, X_test, y_test)

# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## Helper method to print a confusion matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

test_predictions = model.predict(X_test, batch_size=batch_size)
y_test_pred = [np.argmax(x) for x in test_predictions]
cnf_matrix = confusion_matrix(y_test, y_test_pred)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
