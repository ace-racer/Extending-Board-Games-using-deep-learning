from itertools import product

cols = "abcdefgh".split("")
rows = "12345678".split("")
cartesian_product = list(product(cols, rows))
print(cartesian_product)
board_positions = [x[0] + x[1] for x in cartesian_product]
print(board_positions)
chess_pieces = "p"*8 + "n"*2 + "r"*2 + "b"*2 + "k" + "q"
print(chess_pieces)
black_pieces = ["b" + x for x in chess_pieces.split("")]
white_pieces = ["w" + x for x in chess_pieces.split("")]


def generate_dummy_board():


