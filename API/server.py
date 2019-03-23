# import the necessary packages
from PIL import Image
import numpy as np
import flask
import io
import cv2
from flask_cors import CORS
from flask import request

import constants, configurations, utils
from requestprocessor import RequestProcessor
from redisprovider import RedisProvider
from mongodbprovider import MongoDBProvider

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)

global request_processor
global redis_provider
global mongo_provider

@app.route("/digitize_board", methods=["POST"])
def digitize_chess_board():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image_str = flask.request.files["image"].read()

            # Solution taken from https://stackoverflow.com/questions/47515243/reading-image-file-file-storage-object-using-cv2
            # convert string data to numpy array
            npimg = np.fromstring(image_str, np.uint8)

            # convert numpy array to image
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            gameid = flask.request.form["gameid"]
            move_number = flask.request.form["move_number"]

            # NOTE: these values are received as strings
            print(gameid)
            print(move_number)

            if gameid and move_number:
                print(image.shape)
                
                positions_with_pieces = request_processor.process_chess_board_image(move_number, gameid, image)
                mongo_provider.insert_record_with_properties(positions_with_pieces, { constants.SEQUENCE_NUM_STR: 1, constants.TYPE_STR: "combined_prediction", constants.MOVE_NUMBER_STR: move_number, constants.GAME_ID_STR: gameid}, constants.LOGS_COLLECTION)
                current_position_rules_results = request_processor.check_rules(positions_with_pieces)
                mongo_provider.insert_record_with_properties({"rules_violated": current_position_rules_results[0], "rules_violated_details": current_position_rules_results[1]}, {constants.SEQUENCE_NUM_STR: 2, constants.TYPE_STR: "position_rules", constants.MOVE_NUMBER_STR: move_number, constants.GAME_ID_STR: gameid}, constants.LOGS_COLLECTION)
                
                existing_boards = redis_provider.get_value_in_redis(gameid)
                
                # create the existing boards if they do not exist in Redis
                if not existing_boards:
                    existing_boards = {}
                existing_boards[move_number] = positions_with_pieces
                redis_provider.set_value_in_redis(gameid, existing_boards)
                
                data["board"] = existing_boards
                data["success"] = True

                if current_position_rules_results:
                    data["rules_violated"] = current_position_rules_results[0]
                    data["rules_violated_details"] = current_position_rules_results[1]
            else:
                data["message"] = "Game Id and move number are mandatory for the request..."
        else:
            data["message"] = "Chess board image not part of request..."

    print(data)
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/predict_piece", methods=["POST"])
def predict_piece():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            if image:
                image = np.array(image)               
                print(image.shape)
                piece_types_with_confidence = request_processor.predict_piece_type(image)
                data["piece_details"] = piece_types_with_confidence
                data["success"] = True

    return flask.jsonify(data)

@app.route("/predict_color", methods=["POST"])
def predict_color():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            if image:
                image = np.array(image)               
                piece_colors = request_processor.predict_piece_color_empty(image)
                data["piece_details"] = piece_colors
                data["success"] = True

    return flask.jsonify(data)

@app.route("/segment_board", methods=["POST"])
def segment_board():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            if image:
                image = np.array(image)               
                print(image.shape)
                request_processor.segment_chess_board(image)
                data["success"] = True

    return flask.jsonify(data)

@app.route("/add_actual_move", methods=["POST"])
def add_actual_move():
    request_content = request.json
    mongo_provider.insert_record_with_properties(request_content, {constants.TYPE_STR: "actual_move"}, constants.LOGS_COLLECTION)
    data = {"success": True}
    return flask.jsonify(data)

@app.route("/get_logs", methods=["GET"])
def get_logs():
    # get the game ID from the query string
    game_id = request.args.get("gameid")
    print(game_id)
    data = { "success": False }
    log_details = "GameId not present in request"

    if game_id:
        data["success"] = True
        log_details = mongo_provider.retrieve_all_records(constants.LOGS_COLLECTION, {constants.GAME_ID_STR: game_id})

    data["details"] = log_details
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    redis_provider = RedisProvider()
    mongo_provider = MongoDBProvider()
    request_processor = RequestProcessor(mongo_provider)
    app.run(debug=True, use_reloader=False)
