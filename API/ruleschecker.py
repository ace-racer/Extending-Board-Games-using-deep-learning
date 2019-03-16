import constants
import configurations
from collections import defaultdict

class RulesChecker:
    def __init__(self):
        self._piece_wise_rules = {
            "k": {"min": 1, "max": 1, "priority": 0},
            "q": {"min": 0, "max": 1, "priority": 1},
            "p": {"min": 0, "max": 8, "priority": 3},
            "b": {"min": 0, "max": 2, "priority": 2},
            "n": {"min": 0, "max": 2, "priority": 2},
            "r": {"min": 0, "max": 2, "priority": 2},
        }

    def check_piece_numbers(self, pieces_with_positions):
        if pieces_with_positions:
            error_message = ""
            is_error = False

            chess_pieces_counter = defaultdict(int)
            for position in pieces_with_positions:
                # add count for the piece at the provided position
                chess_pieces_counter[pieces_with_positions[position]] += 1

            for chess_piece in chess_pieces_counter:
                piece_name_lower = chess_piece[-1].lower()
                piece_rule = self._piece_wise_rules.get(piece_name_lower)
                if piece_rule["min"] == piece_rule["max"]:
                    if chess_pieces_counter[chess_piece] != piece_rule["min"]:
                        is_error = True
                        error_message = "Exactly {0} pieces expected for {1}".format(piece_rule["min"], chess_piece)
                        break
                else:
                    if chess_pieces_counter[chess_piece] > piece_rule["max"]:
                        is_error = True
                        error_message = "Atmost {0} pieces expected for {1}".format(piece_rule["max"], chess_piece)
                        break

            return is_error, error_message

                
