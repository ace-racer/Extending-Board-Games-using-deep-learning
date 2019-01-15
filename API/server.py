# import the necessary packages
from PIL import Image
import numpy as np
import flask
import io

import constants, configurations, utils
from requestprocessor import RequestProcessor
from redisprovider import RedisProvider

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
global request_processor
global redis_provider

@app.route("/digitize_board", methods=["POST"])
def digitize_chess_board():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            gameid = flask.request.form["gameid"]
            move_number = flask.request.form["move_number"]

            print(gameid)
            print(move_number)

            if gameid and move_number and image:
                image = np.array(image)
                move_number = int(move_number)
                print(image.shape)
                positions_with_pieces = request_processor.process_chess_board_image(move_number, gameid, image)
                existing_boards = redis_provider.get_value_in_redis(gameid)
                
                # create the existing boards if they do not exist in Redis
                if not existing_boards:
                    existing_boards = {}
                existing_boards[move_number] = positions_with_pieces
                redis_provider.set_value_in_redis(gameid, existing_boards)
                
                data["board"] = existing_boards
                data["success"] = True
            else:
                data["message"] = "Game Id and move number are mandatory for the request..."
        else:
            data["message"] = "Chess board image not part of request..."


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

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    request_processor = RequestProcessor()
    redis_provider = RedisProvider()
    app.run()
