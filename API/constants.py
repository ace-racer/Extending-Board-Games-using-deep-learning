class_names = ["wB", "wK", "wN", "wP", "wQ", "wR", "bB", "bK", "bN", "bP", "bQ", "bR", "empty"]
full_class_names = ["white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook", "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook", "empty"]
num_output_classes = len(class_names)

#Mongo DB and collections
database_name="Extendingboardgames"
request_chessboard_details_collection="RequestChessBoardDetails"
segmented_chessboard_details_collection="SegmentedChessBoardDetails"