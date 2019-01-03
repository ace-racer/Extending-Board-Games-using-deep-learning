from itertools import product
import numpy as np
import random

cols = list("abcdefgh")
rows = list("12345678")
cartesian_product = list(product(cols, rows))
print(cartesian_product)
board_positions = [x[0] + x[1] for x in cartesian_product]
print(board_positions)
chess_pieces = list("p"*8 + "n"*2 + "r"*2 + "b"*2 + "k" + "q")
print(chess_pieces)
black_pieces = ["b" + x for x in chess_pieces]
white_pieces = ["w" + x for x in chess_pieces]
all_pieces = black_pieces
all_pieces.extend(white_pieces)
print("All pieces...")
print(all_pieces)

def generate_dummy_board(num_pieces = 32):
    chosen_positions = random.sample(board_positions, num_pieces)
    chosen_pieces = random.sample(all_pieces, num_pieces)
    return dict(zip(chosen_positions, chosen_pieces))



