class_names = ["wB", "wK", "wN", "wP", "wQ", "wR", "bB", "bK", "bN", "bP", "bQ", "bR", "empty"]
full_class_names = ["white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook", "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook", "empty"]
num_output_classes = len(class_names)
InceptionV3_Image_Dimension = (299, 299)

#Mongo DB and collections
database_name="Extendingboardgames"
request_chessboard_details_collection="RequestChessBoardDetails"
segmented_chessboard_details_collection="SegmentedChessBoardDetails"
NUMBER_TO_CATEGORY_MAPPING = {0: "b", 1: "w", 2: "empty"}
NUMBER_TO_CHESS_PIECE_TYPE_MAPPING = [ "P", "B", "N", "R", "Q", "K" ]
LOGS_COLLECTION = "logs"
TYPE_STR = "type"
MOVE_NUMBER_STR = "move_number"
GAME_ID_STR = "game_id"