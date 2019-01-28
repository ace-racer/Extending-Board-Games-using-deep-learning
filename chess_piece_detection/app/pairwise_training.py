import numpy as np
import keras
import os
import itertools
import random
from collections import Counter, defaultdict
from itertools import product, combinations
import math

import cv2
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import siamese_network

#samples_per_type = {"b": 30, "n": 25, "k": 25, "p": 35, "q": 25, "r": 35}
samples_per_type = {"b": 3, "n": 2, "k": 2, "p": 3, "q": 2, "r": 3}

# training parameters
IMAGE_SIZE = (100, 100)
CHECKPOINTS_LOCATION = "weights"
LOGS_LOCATION = "logs"
BATCH_SIZE = 32
NUM_EPOCHS = 50

# change as required
IMAGES_LOCATION = "Chess-Pieces-Data/crawled_1901/"
#IMAGES_LOCATION = "H:\\AR-ExtendingOnlineGames\\crawled_chess_piece_images"

if not os.path.exists(CHECKPOINTS_LOCATION):
    os.makedirs(CHECKPOINTS_LOCATION)

if not os.path.exists(LOGS_LOCATION):
    os.makedirs(LOGS_LOCATION)

def generate_paired_instances_by_ratio(folder_name, total_instances = 10000, different_records_ratio = 0.5):
    
    data = []
    label_values = []
    for type_name in samples_per_type:
        piece_type_folder = os.path.join(folder_name, type_name)
        for f in (os.listdir(piece_type_folder)):
            img_file_loc = os.path.join(piece_type_folder, f)
            data.append(img_file_loc)
            label_values.append(type_name)
    
    num_categories = 6

    # Get the counts of the individual labels
    label_counts = Counter(label_values)
    
    # Get the label indices in the original data read from the file
    label_indices = defaultdict(list)
    for itr, val in enumerate(label_values):
        label_indices[val].append(itr)
    
    num_same_items_per_category = int(math.ceil(np.sqrt((( 1- different_records_ratio ) * total_instances) / num_categories)))
    num_different_items_per_category = int(math.ceil(np.sqrt((2 * different_records_ratio * total_instances)/(num_categories * (num_categories - 1)))))
    print("Num same items per category: " + str(num_same_items_per_category))
    print("Num different items per category: " + str(num_different_items_per_category))

    most_common_categories = [x for x, _ in label_counts.most_common(num_categories)]
    print("Most common categories...")
    print(most_common_categories)
    
    pairwise_indices_same_items = []
    for label in most_common_categories:
        required_indices = label_indices[label][:num_same_items_per_category]
        similar_item_index_pairs = list(product(required_indices, required_indices))
        pairwise_indices_same_items.extend(similar_item_index_pairs)

    pairwise_indices_different_items = []
    category_pairs = combinations(most_common_categories, 2)

    for cat1, cat2 in category_pairs:
        category1_indices = label_indices[cat1][:num_different_items_per_category]
        category2_indices = label_indices[cat2][:num_different_items_per_category]
        different_items_index_pairs = list(product(category1_indices, category2_indices))
        pairwise_indices_different_items.extend(different_items_index_pairs)

    print("Num same category pairs: " + str(len(pairwise_indices_same_items)))
    print("Num different category pairs: " + str(len(pairwise_indices_different_items)))

    instances_with_labels = []
    for idx1, idx2 in pairwise_indices_same_items:
        label = int(label_values[idx1] == label_values[idx2])
        
        img1 = cv2.imread(data[idx1])
        img1 = cv2.resize(img1, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        img2 = cv2.imread(data[idx2])
        img2 = cv2.resize(img2, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        instances_with_labels.append((img1, img2, label))

    for idx1, idx2 in pairwise_indices_different_items:
        label = int(label_values[idx1] == label_values[idx2])

        img1 = cv2.imread(data[idx1])
        img1 = cv2.resize(img1, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        img2 = cv2.imread(data[idx2])
        img2 = cv2.resize(img2, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        instances_with_labels.append((img1, img2, label))

    random.shuffle(instances_with_labels)
    instances = np.array([[x[0], x[1]] for x in instances_with_labels])
    labels = np.array([x[2] for x in instances_with_labels])

    return instances, labels

X_train_original = []
y_train_original = []


training_images = os.path.join(IMAGES_LOCATION, "train")

X_train_original, y_train_original = generate_paired_instances_by_ratio(training_images)

X_train_original = np.array(X_train_original)
X_train_original = X_train_original.astype('float32')
X_train_original /= 255
y_train_original = np.array(y_train_original)

print(X_train_original.shape)
print(y_train_original.shape)

# split into train and validation splits
X_train, X_test, y_train, y_test = train_test_split(X_train_original, y_train_original, test_size=0.25, random_state=42, stratify = y_train_original)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train_left = X_train[:, 0, ...]
X_train_right = X_train[:, 1, ...]
print(X_train_left.shape)
print(X_train_right.shape)

X_test_left = X_test[:, 0, ...]
X_test_right = X_test[:, 1, ...]


filepath = os.path.join(CHECKPOINTS_LOCATION, "siamese.hdf5")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='max')

tensorboard = TensorBoard(log_dir=LOGS_LOCATION, histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [checkpoint, earlystop, tensorboard]


model = siamese_network.siamese_net
X_train_instances = [X_train_left, X_train_right]
X_test_instances = [X_test_left, X_test_right]

hist = model.fit(X_train_instances, y_train, shuffle=True, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS, verbose=1, validation_data=(X_test_instances, y_test), callbacks=callbacks_list)


