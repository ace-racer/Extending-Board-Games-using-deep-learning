# import the necessary packages
from PIL import Image
import numpy as np
import io
import os

import constants
import configurations
import utils
from trained_models_invoker import TrainedModelsInvoker
from mongodbprovider import MongoDBProvider

class ChessPieceRecognition(TrainedModelsInvoker):
    def __init__(self, mongo_db_provider = None):
        print("loading models...")
        self._mongo_db_provider = mongo_db_provider
        self._six_class_cnn_model = self.load_6_class_classifier()
        self._three_class_cnn_model = self.load_3_class_classifier()

    def predict_classes_for_segmented_images(self, segmented_images_with_positions, move_number, game_id):
        print("Performing predictions for segmented images...")
        segmented_images = [x["image"] for x in segmented_images_with_positions]
        positions = [x["position"] for x in segmented_images_with_positions]

        processed_chess_piece_images, predicted_colors = self.predict_color_empty_for_image(segmented_images)
        assert(len(predicted_colors) == len(segmented_images))
        self._mongo_db_provider.insert_record_with_properties({"predicted_colors": predicted_colors}, {constants.SEQUENCE_NUM_STR: 3, constants.TYPE_STR: "color_empty_prediction", constants.MOVE_NUMBER_STR: move_number, constants.GAME_ID_STR: game_id}, constants.LOGS_COLLECTION)

        predicted_pieces_with_positions = self.predict_pieces_given_colors(processed_chess_piece_images, predicted_colors, positions)
        return predicted_pieces_with_positions

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
        return processed_chess_piece_images, predictions_str


    def predict_pieces_given_colors(self, processed_chess_piece_images, predicted_colors, positions):
        assert(len(predicted_colors) == len(positions))
        details = {}
        
        predicted_piece_types = self._six_class_cnn_model.predict(processed_chess_piece_images, batch_size=64)
        predicted_piece_types = [np.argmax(x) for x in predicted_piece_types]

        for itr, predicted_color_empty in enumerate(predicted_colors):
            if predicted_color_empty != "empty":
                # for the positions that are not identified as empty predict the type for the image
                details[positions[itr]] = predicted_color_empty + constants.NUMBER_TO_CHESS_PIECE_TYPE_MAPPING[predicted_piece_types[itr]]

        print(details)
        return details




