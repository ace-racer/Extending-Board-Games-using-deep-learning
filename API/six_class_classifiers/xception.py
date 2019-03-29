import keras
from keras.layers import Input, Dense, Flatten,MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model, Sequential


from keras.optimizers import SGD,Adam


import numpy.random as rng
from keras.applications.xception import Xception

import numpy as np

import utils
import configurations
import constants

IMAGE_DIMS = (80, 80)
IMAGE_NUM_CHANNELS = 3

def load_6_class_classifier():
    required_input_shape = (*IMAGE_DIMS, IMAGE_NUM_CHANNELS)

    base_model = Xception(include_top=False, weights='imagenet', input_shape=required_input_shape)
    # add a global spatial average pooling layer

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a softmax layer
    predictions = Dense(6, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # load the model weights
    model.load_weights(configurations.six_class_xception_model_location)
                        
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

def process_images_for_prediction(chess_piece_images):
    """Process the image as required by the xception model"""
    processed_chess_piece_images = []
    for chess_piece_image in chess_piece_images:
        processed_chess_piece_image = utils.resize_given_image(chess_piece_image, IMAGE_DIMS)
        processed_chess_piece_images.append(processed_chess_piece_image)

    processed_chess_piece_images = np.array(processed_chess_piece_images)
    processed_chess_piece_images = processed_chess_piece_images.astype('float32')
    processed_chess_piece_images /= 255

    return processed_chess_piece_images