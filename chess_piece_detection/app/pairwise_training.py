import numpy as np
import keras
import os
import itertools
import random

import cv2
from sklearn.model_selection import train_test_split

IMAGES_LOCATION = "H:\\AR-ExtendingOnlineGames\\crawled_chess_piece_images"
IMAGE_SIZE = (200, 200)

X_train_original = []
y_train_original = []


training_images = os.path.join(IMAGES_LOCATION, "train")
validation_images = os.path.join(IMAGES_LOCATION, "test")

samples_per_type = {"b": 30, "n": 25, "k": 25, "p": 35, "q": 25, "r": 35}
files_with_labels = []

for type_name in samples_per_type:
    piece_type_folder = os.path.join(training_images, type_name)
    for idx, f in enumerate(os.listdir(piece_type_folder)):
        if idx >= samples_per_type[type_name]:
            break

        img_file_loc = os.path.join(piece_type_folder, f)
        files_with_labels.append((img_file_loc, type_name))


random.shuffle(files_with_labels)
# print(files_with_labels)

cartesian_product = itertools.product(files_with_labels, files_with_labels)
# print(cartesian_product)

for item1, item2 in cartesian_product:

    img1 = cv2.imread(item1[0])
    img1 = cv2.resize(img1, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    img2 = cv2.imread(item2[0])
    img2 = cv2.resize(img2, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    label = int(item1[1] == item2[1])
    X_train_original.append([img1, img2])
    y_train_original.append(label)

X_train_original = np.array(X_train_original)
y_train_original = np.array(y_train_original)

print(X_train_original.shape)
print(y_train_original.shape)

# split into train and validation splits
X_train, X_test, y_train, y_test = train_test_split(X_train_original, y_train_original, test_size=0.33, random_state=42, stratify = y_train_original)

