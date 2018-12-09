# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from PIL import Image
import numpy as np
import flask
import io
import os

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

model_folder_name = "../../models/09_12"
model_name = "chess_pieces_inceptionv3_p2.hdf5"

class_names = ["bishop", "king", "knight", "pawn", "queen", "rook", "empty"]
class_names_reverse_mappings = {"bishop": 0, "king": 1, "knight":2, "pawn":3, "queen":4, "rook":5, "empty":6}
num_output_classes = len(class_names)

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    # create the base pre-trained model
    inception_v3_model = InceptionV3(include_top=False)

    # add a global spatial average pooling layer
    x = inception_v3_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_output_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inception_v3_model.input, outputs=predictions)
    model.load_weights(os.path.join(model_folder_name, model_name))

def prepare_image(image, dimensions):
    # basic pre-processing of the images
    img = image.resize(dimensions)
    x = image.img_to_array(img)

    # since only single image so expand dims
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # return the processed image
    return image

def decode_predictions(predictions):
    # outputs a batch of predictions
    predicted_class_id = [np.argmax(x) for x in predictions][0]
    predicted_class_probability = predictions[predicted_class_id]
    return class_names[predicted_class_id], predicted_class_probability

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, dimensions=(299, 299))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = decode_predictions(preds)
            data["type"] = results[0]
            data["probability"] = results[1]


            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()