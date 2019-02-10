# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
from PIL import Image
import numpy as np
import io
import os

import constants
import configurations
import utils

class ChessPieceRecognition:
    def __init__(self):
        # self._model = self.load_model()
        self._model = None
        self._three_class_cnn_model = self.load_3_class_cnn_model()

    def load_model(self):
        # load the model structure
        print("Loading model for inference...")
        inception_v3_model = InceptionV3(include_top=False)
        x = inception_v3_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(constants.num_output_classes, activation='softmax')(x)
        model = Model(inputs=inception_v3_model.input, outputs=predictions)

        # load the model weights
        model.load_weights(os.path.join(configurations.model_folder_name, configurations.model_name))
        return model

    def load_3_class_cnn_model(self):
        """Load the 3 class CNN model for inference"""
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



    def prepare_images(self, input_images, dimensions):
        prepared_input_images = []
        for input_image in input_images:
            # print("Shape: " + str(input_image.shape))
            # basic pre-processing of the images
            resized_input_image = utils.upsize_image(input_image, dimensions)
            processed_image = preprocess_input(resized_input_image)
            prepared_input_images.append(processed_image)
        
        prepared_input_images = np.array(prepared_input_images)
        print(prepared_input_images.shape)

        # return the prepared images
        return prepared_input_images

    def decode_predictions_for_segmented_images(self, predictions, positions):
        # outputs a batch of predictions
        print("Predictions...")
        #print(predictions)
        positions_with_predictions = {}
        for idx, prediction in enumerate(predictions):
            predicted_class_id = np.argmax(prediction)
            positions_with_predictions[positions[idx]] = constants.class_names[predicted_class_id]

        return positions_with_predictions

    def predict_classes_for_segmented_images(self, segmented_images_with_positions):
        print("Performing predictions for segmented images...")
        segmented_images = [x["image"] for x in segmented_images_with_positions]
        positions = [x["position"] for x in segmented_images_with_positions]
        prepared_segmented_images = self.prepare_images(segmented_images, constants.InceptionV3_Image_Dimension)
        preds = self._model.predict(prepared_segmented_images, batch_size=64)
        if preds is not None:
            return self.decode_predictions_for_segmented_images(preds, positions)
        else:
            return None

    def predict_class_for_images(self, chess_piece_images):
        prepared_images = self.prepare_images(chess_piece_images, constants.InceptionV3_Image_Dimension)
        preds = self._model.predict(prepared_images, batch_size=64)
        if preds is not None:
            predictions_with_confidence = []
            for pred in preds:
                predicted_class_id = np.argmax(pred)
                predicted_class_confidence = str(round(pred[predicted_class_id], 5))
                predictions_with_confidence.append({"type": constants.class_names[predicted_class_id], "confidence": predicted_class_confidence})
            
            return predictions_with_confidence
        else:
            return None

    def predict_color_empty_for_image(self, chess_piece_images):
        processed_chess_piece_images = []
        for chess_piece_image in chess_piece_images:
            processed_chess_piece_image = utils.process_image_three_class_cnn(chess_piece_image)
            processed_chess_piece_image = processed_chess_piece_image[..., np.newaxis]
            processed_chess_piece_images.append(processed_chess_piece_image)

        processed_chess_piece_images = np.array(processed_chess_piece_images)
        processed_chess_piece_images = processed_chess_piece_images.astype('float32')
        processed_chess_piece_images /= 255

        predictions = self._three_class_cnn_model.predict(processed_chess_piece_images, batch_size=64)
        prediction_values = [np.argmax(x) for x in predictions]

        predictions_str = [constants.NUMBER_TO_CATEGORY_MAPPING[x] for x in prediction_values]
        return predictions_str




