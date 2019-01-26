import numpy as np
import keras
import os
import itertools
import random

import cv2

import siamese_network

# change as required
IMAGES_LOCATION = "H:\\AR-ExtendingOnlineGames\\crawled_chess_piece_images"
# training parameters
IMAGE_SIZE = (100, 100)
CHECKPOINTS_LOCATION = "H:\\AR-ExtendingOnlineGames\\ExtendingGames_Code\\models\\siamese\\weights\\weights"
REPRESENTATIVE_IMAGES = "H:\\AR-ExtendingOnlineGames\\crawled_chess_piece_images_git\\crawled_1901\\repr"

test_file_location = "H:\\AR-ExtendingOnlineGames\\crawled_chess_piece_images_git\\crawled_1901\\repr\\b.jpg"

model = siamese_network.siamese_net
model.load_weights(os.path.join(CHECKPOINTS_LOCATION, "siamese.hdf5"))

X_test = []

# create the test data
for repr_image in os.listdir(REPRESENTATIVE_IMAGES):
    repr_image_loc = os.path.join(REPRESENTATIVE_IMAGES, repr_image)
    img1 = cv2.imread(repr_image_loc)
    img1 = cv2.resize(img1, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    img2 = cv2.imread(test_file_location)
    img2 = cv2.resize(img2, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    X_test.append([img1, img2])

X_test = np.array(X_test)
X_test = X_test.astype('float32')
X_test /= 255
print(X_test.shape)

X_test_left = X_test[:, 0, ...]
X_test_right = X_test[:, 1, ...]
X_test_instances = [X_test_left, X_test_right]

test_predictions = model.predict(X_test_instances)

print(test_predictions)