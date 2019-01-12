from redisprovider import RedisProvider
from mongodbprovider import MongoDBProvider
import utils
import constants
import configurations

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
        pass
