from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras import optimizers


import constants
import configurations


class TrainedModelsInvoker:
    def load_3_class_classifier(self):
        """Load the 3 class CNN model for inference"""
        required_input_shape = (*configurations.CHESS_BLOCK_IMAGE_SIZE, 1)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='valid', input_shape=required_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.6))
        
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.summary()

        
        # load the model weights
        model.load_weights(configurations.three_class_cnn_model_location)
                            
        adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        return model

    def load_6_class_classifier(self):
        """
            Load the 6 class classifer model to infer the type of the chess piece
        """

        required_input_shape = (*configurations.CHESS_BLOCK_IMAGE_SIZE, 1)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='valid', input_shape=required_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.5))

        model.add(Flatten())
        
        model.add(Dense(2048))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.6))
        
        model.add(Dense(6))
        model.add(Activation('softmax'))
        model.summary()

        # load the model weights
        model.load_weights(configurations.six_class_cnn_model_location)
                            
        adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        return model


