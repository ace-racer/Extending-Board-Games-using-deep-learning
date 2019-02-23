from redisprovider import RedisProvider
from mongodbprovider import MongoDBProvider
from chess_board_segmentation import ChessBoardSegmentation
from chess_piece_recognition import ChessPieceRecognition
import utils
import constants
import configurations
import cv2
import numpy as np

import os

class RequestProcessor:
    def __init__(self):
        self._mongo_db_provider = MongoDBProvider()
        self._redis_provider = RedisProvider()
        self._chess_pieces_recognizer = ChessPieceRecognition()
        self._chess_board_segmenter = ChessBoardSegmentation()


    def process_chess_board_image(self, move_number, game_id, chess_board_image):
        
        # Step 0: Resize and perform any preprocessing on the incoming image
        # chess_board_image = utils.downsize_image(chess_board_image, configurations.REQUIRED_CHESS_BOARD_DIMENSION)

        # Step 1: serialize the incoming image and store request object for persistence
        serialized_chess_board_image = utils.base64_encode_image(chess_board_image)
        request_obj = {"move_number": move_number, "game_id": game_id, "chess_board_image": serialized_chess_board_image}
        #self._mongo_db_provider.insert_record(request_obj, constants.request_chessboard_details_collection)

        # Step 2: Segment the chess board image and get a list of images
        segmented_images = self._chess_board_segmenter.segment_board_corners_provided(chess_board_image, is_file=False)
        serialized_segmented_images = {x["position"]: utils.base64_encode_image(x["image"]) for x in segmented_images}
        
        # Step 3: Store the segmented images in the segmented images collection after serialization
        segmented_images_obj = {"move_number": move_number, "game_id": game_id, "segmented_images": serialized_segmented_images}
        #self._mongo_db_provider.insert_record(segmented_images_obj, constants.segmented_chessboard_details_collection)

        # Step 4: Retrieve the segmented images for the last move for the same game
        previous_move_number = move_number - 1

        # By default classify all the segmented images
        segmented_images_for_classification = segmented_images

        if configurations.APP_FLAGS["SEND_DELTA_ONLY"] and previous_move_number >= 0:
            print("Previous move number is {0}".format(previous_move_number))
            last_move_segmented_images_query = {"move_number": previous_move_number, "game_id": game_id}
            previous_move_segmented_images_obj = self._mongo_db_provider.retrieve_record(constants.segmented_chessboard_details_collection, last_move_segmented_images_query)

            if previous_move_segmented_images_obj:
                print("Previous move and segmented images for the previous move exists")
                previous_move_segmented_images = previous_move_segmented_images_obj.get("segmented_images")

                if previous_move_segmented_images:
                    required_positions = []
                    segmented_images_for_classification = []

                    # Step 5: Compare the segmented images for the last move to find differences
                    # compare the segmented images using hash values and add only required images to `segmented_images_for_classification`
                    # TODO: replace with hash values for faster comparison
                    for pos in previous_move_segmented_images:
                        required_segment_current_board = serialized_segmented_images.get(pos)
                        if required_segment_current_board:
                            if required_segment_current_board != previous_move_segmented_images[pos]:
                                print("The position {0} has changed".format(pos))
                                required_positions.append(pos)
                            else:
                                print("The position {0} has not changed...".format(pos))
                        else:
                            print("The position {0} does not exist for the current board...".format(pos))

                    segmented_images_for_classification = [x for x in segmented_images if x["position"] in required_positions]

        # perform detection only if the segmented images list is not empty
        if len(segmented_images_for_classification) > 0:
            # Step 6: Classify the segmented images for classification
            return self._chess_pieces_recognizer.predict_classes_for_segmented_images(segmented_images_for_classification)
        return {}


    def predict_piece_type(self, piece_image):
        images_for_recognition = np.array([piece_image])
        processed_chess_piece_images, predictions_str = self._chess_pieces_recognizer.predict_color_empty_for_image(images_for_recognition)
        positions = list(range(images_for_recognition.shape[0]))
        return self._chess_pieces_recognizer.predict_pieces_given_colors(processed_chess_piece_images, predictions_str, positions)

    def segment_chess_board(self, chess_board_image):
        print(chess_board_image.shape)
        return self._chess_board_segmenter.segment_board_corners_provided(chess_board_image, is_file=False)

    def predict_piece_color_empty(self, piece_image):
        images_for_recognition = np.array([piece_image])
        return self._chess_pieces_recognizer.predict_color_empty_for_image(images_for_recognition)

        


