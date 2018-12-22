# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from PIL import Image
import numpy as np
import io
import os

import constants
import configurations


def load_model():
    global model

    # create the base pre-trained model
    inception_v3_model = InceptionV3(include_top=False)
    
    num_output_classes = len(constants.class_names)

    # add a global spatial average pooling layer
    x = inception_v3_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_output_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inception_v3_model.input, outputs=predictions)
    model.load_weights(os.path.join(configurations.model_folder_name, configurations.model_name))

def prepare_image(input_image, dimensions):
    # basic pre-processing of the images
    resized_input_image = input_image.resize(dimensions)
    x = image.img_to_array(resized_input_image)

    # since only single image so expand dims
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # return the processed image
    return x

def decode_predictions(predictions):
    # outputs a batch of predictions
    print("Predictions...")
    print(predictions)
    required_prediction = predictions[0]
    predicted_class_id = np.argmax(required_prediction)
    predicted_class_probability = required_prediction[predicted_class_id]
    return constants.class_names[predicted_class_id], predicted_class_probability

def make_prediction(img):
    preds = model.predict(img)
    if preds is not None:
        return decode_predictions(preds)
    else:
        return None
