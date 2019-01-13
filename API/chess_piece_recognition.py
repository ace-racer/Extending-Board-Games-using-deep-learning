# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from PIL import Image
import numpy as np
import io
import os

import constants
import configurations
import utils

class ChessPieceRecognition:
    def __init__(self):
        self._model = self.load_model()

    def load_model(self):
        # load the model structure
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
        preds = self._model.predict(prepared_segmented_images)
        if preds is not None:
            return self.decode_predictions_for_segmented_images(preds, positions)
        else:
            return None

    def predict_class_for_images(self, chess_piece_images):
        prepared_images = self.prepare_images(chess_piece_images, constants.InceptionV3_Image_Dimension)
        preds = self._model.predict(prepared_images)
        if preds is not None:
            predictions_with_confidence = []
            for pred in preds:
                predicted_class_id = np.argmax(pred)
                predicted_class_confidence = pred[predicted_class_id]
                predictions_with_confidence.append({"type": constants.class_names[predicted_class_id], "confidence": predicted_class_confidence})
            
            return predictions_with_confidence
        else:
            return None

