from redisprovider import RedisProvider
from mongodbprovider import MongoDBProvider
from chess_board_segmentation import ChessBoardSegmentation
import utils
import constants
import configurations

import os

class RequestProcessor:
    def __init__(self):
        self._mongo_db_provider = MongoDBProvider()
        self._redis_provider = RedisProvider()

    def process_chess_board_image(self, move_number, game_id, chess_board_image):

        # Step 1: serialize the incoming image and store request object for persistence
        serialized_chess_board_image = utils.base64_encode_image(chess_board_image)
        request_obj = {"move_number": move_number, "game_id": game_id, "chess_board_image": serialized_chess_board_image}
        self._mongo_db_provider.insert_record(request_obj, constants.request_chessboard_details_collection)

        # Step 2: Segment the chess board image and get a list of images
        chess_board_segmenter = ChessBoardSegmentation()
        board = chess_board_segmenter.find_board(chess_board_image, is_file=False)
        cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION, "board_image_cropped.jpg"), board)
        segmented_images = chess_board_segmenter.split_board(board)
        serialized_segmented_images = [utils.base64_encode_image(x) for x in segmented_images]
        
        # Step 3: Store the segmented images in the segmented images collection after serialization
        segmented_images_obj = {"move_number": move_number, "game_id": game_id, "segmented_images": serialized_segmented_images}
        self._mongo_db_provider.insert_record(segmented_images_obj, constants.segmented_chessboard_details_collection)

        # Step 4: Retrieve the segmented images for the last move for the same game
        previous_move_number = move_number - 1

        # By default classify all the segmented images
        segmented_images_for_classification = segmented_images

        if previous_move_number >= 0:
            print("Previous move number is {0}".format(previous_move_number))
            last_move_segmented_images_query = {"move_number": previous_move_number, "game_id": game_id}
            previous_move_segmented_images = self._mongo_db_provider.retrieve_record(constants.segmented_chessboard_details_collection, last_move_segmented_images_query)

            if previous_move_segmented_images and previous_move_segmented_images.get("segmented_images"):
                print("Previous move and segmented images for the previous move exists")

                # Step 5: Compare the segmented images for the last move to find differences
                # TODO: compare the segmented images using hash values and add only required images to `segmented_images_for_classification`

        # Step 6: Classify the segmented images for classification
        for segmeted_image in segmented_images_for_classification:
            pass
