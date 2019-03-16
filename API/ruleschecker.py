import constants
import configurations

class RulesChecker:
    def __init__(self):
        self._piece_wise_rules = {
            "k": {"min": 1, "max": 1, "priority": 0},
            "q": {"min": 0, "max": 1, "priority": 1},
            "p": {"min": 0, "max": 8, "priority": 3},
            "b": {"min": 0, "max": 2, "priority": 2},
            "n": {"min": 0, "max": 2, "priority": 2},
            "r": {"min": 0, "max": 2, "priority": 2}
        }

        self._piece_name_mapping = dict(zip(constants.class_names, constants.full_class_names))

    def check_piece_numbers(self, pieces_with_positions):
        if pieces_with_positions:
            error_message = ""
            is_error = False

            # set an arbitrary max value
            current_violation_level = 100

            # initialize all the pieces with 0's
            chess_pieces_counter = {x:0 for x in constants.class_names if x != "empty"}

            for position in pieces_with_positions:
                # add count for the piece at the provided position
                chess_pieces_counter[pieces_with_positions[position]] += 1

            for chess_piece in chess_pieces_counter:
                piece_name_lower = chess_piece[-1].lower()
                piece_rule = self._piece_wise_rules.get(piece_name_lower)
                if piece_rule["min"] == piece_rule["max"]:
                    if chess_pieces_counter[chess_piece] != piece_rule["min"]:
                        if piece_rule["priority"] < current_violation_level:
                            current_violation_level = piece_rule["priority"]
                            is_error = True
                            error_message = "Exactly {0} pieces expected for {1}".format(piece_rule["min"], self._piece_name_mapping.get(chess_piece))
                        
                else:
                    if chess_pieces_counter[chess_piece] > piece_rule["max"]:
                        if piece_rule["priority"] < current_violation_level:
                            current_violation_level = piece_rule["priority"]
                            is_error = True
                            error_message = "Atmost {0} pieces expected for {1}".format(piece_rule["max"], self._piece_name_mapping.get(chess_piece))
                        

            return is_error, error_message

                
