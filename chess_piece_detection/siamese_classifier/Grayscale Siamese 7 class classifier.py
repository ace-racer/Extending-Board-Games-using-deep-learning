import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import os
import random
from collections import Counter, defaultdict
from itertools import product, combinations
import math
import cv2
from sklearn.model_selection import train_test_split

random.seed(42)

import keras
from keras.layers import Input, Conv2D, Lambda, average, Dense, Flatten,MaxPooling2D, BatchNormalization, Dropout, Activation, Subtract, subtract
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy.random as rng

IMAGE_SIZE = (70, 70)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def process_image(image_location, params):
    """
        Given the image location, process the image
    """
    # print(image_location)
    
    image = cv2.imread(image_location)
    
    if image.shape[0] != IMAGE_SIZE[0] or image.shape[1] != IMAGE_SIZE[1]:
        # print("Resizing the image: {0}".format(image_location))
        resized_image = cv2.resize(image, IMAGE_SIZE, interpolation = cv2.INTER_AREA)
    else:
        resized_image = image
    
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    if params["invert"]:
            gray = cv2.bitwise_not(gray)

    # edges = auto_canny(gray)
    # print(edges.shape)
    
    
    # assert(denoised != edges)
    # weighted_sum = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
    gray = gray[..., np.newaxis]
       
    return gray


"""
Initial code from: https://sorenbouma.github.io/blog/oneshot/

"""

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = *IMAGE_SIZE, 1
print(input_shape)

left_input = Input(input_shape)
right_input = Input(input_shape)

#build convnet to use in each siamese 'leg'
convnet = Sequential()

convnet.add(Conv2D(32,(5,5),input_shape=input_shape, kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(BatchNormalization())
convnet.add(Activation('relu'))
convnet.add(MaxPooling2D())

convnet.add(Conv2D(64,(4,4), kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(BatchNormalization())
convnet.add(Activation('relu'))
convnet.add(MaxPooling2D())

convnet.add(Conv2D(128,(4,4), kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(BatchNormalization())
convnet.add(Activation('relu'))
convnet.add(Flatten())
convnet.add(Dropout(0.4))
convnet.add(Dense(1024,activation="relu",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

#encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

#merge two encoded inputs with the average
both = subtract([encoded_l,encoded_r])
# both = K.abs(both)
# both = Dense(256, activation='relu')(both)
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)


optimizer = Adam(0.0005)

siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

print(siamese_net.count_params())
print(siamese_net.summary())




#samples_per_type = {"b": 30, "n": 25, "k": 25, "p": 35, "q": 25, "r": 35}
# samples_per_type = {"b": 3, "n": 2, "k": 2, "p": 3, "q": 2, "r": 3, "empty": 4}
type_locations = {"b": ["bb", "wb"], "n": ["bn", "wn"], "k": ["bk", "wk"], "p": ["bp", "wp"], "q": ["bq", "wq"], "r": ["br", "wr"], "empty": ["empty"]}

CHECKPOINTS_LOCATION = "weights"
LOGS_LOCATION = "logs"
BATCH_SIZE = 32
NUM_EPOCHS = 50


if not os.path.exists(CHECKPOINTS_LOCATION):
    os.makedirs(CHECKPOINTS_LOCATION)

if not os.path.exists(LOGS_LOCATION):
    os.makedirs(LOGS_LOCATION)

def generate_paired_instances_by_ratio(folder_location, total_instances = 6000, different_records_ratio = 0.5):
    
    data = []
    label_values = []
    for type_name in type_locations:
        for folder_name in type_locations[type_name]:
            piece_type_folder = os.path.join(folder_location, folder_name)
            for f in (os.listdir(piece_type_folder)):
                if f.endswith(".jpg"):
                    params = { "invert": False }

                    if folder_name[0] == "w":
                        params["invert"] = True        
       
                    img_file_loc = os.path.join(piece_type_folder, f)
                    data.append((img_file_loc, params))
                    label_values.append(type_name)
    
    num_categories = len(type_locations)
    print("Num categories: " + str(num_categories))

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
        label = 1
        
        img1 = process_image(*data[idx1])
        img2 = process_image(*data[idx2])
        
        instances_with_labels.append((img1, img2, label))

    for idx1, idx2 in pairwise_indices_different_items:
        label = 0

        img1 = process_image(*data[idx1])
        img2 = process_image(*data[idx2])

        instances_with_labels.append((img1, img2, label))

    random.shuffle(instances_with_labels)
    instances = np.array([[x[0], x[1]] for x in instances_with_labels])
    labels = np.array([x[2] for x in instances_with_labels])

    return instances, labels



# change as required
IMAGES_LOCATION = "C:\\Users\\issuser\\Desktop\\ExtendingBoardGamesOnline\\data\\Chess ID Public Data"

X_train_original = []
y_train_original = []


training_images = os.path.join(IMAGES_LOCATION, "train")

X_train_original, y_train_original = generate_paired_instances_by_ratio(training_images, 10000)

X_train_original = np.array(X_train_original)
y_train_original = np.array(y_train_original)

print(X_train_original.shape)
print(y_train_original.shape)

# split into train and validation splits
# X_train, X_test, y_train, y_test = train_test_split(X_train_original, y_train_original, test_size=0.25, random_state=42, stratify = y_train_original)

X_train = X_train_original
y_train = y_train_original

test_images = os.path.join(IMAGES_LOCATION, "test")
X_test, y_test = generate_paired_instances_by_ratio(test_images, 2000)

X_test = np.array(X_test)
y_test = np.array(y_test)

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


filepath = os.path.join(CHECKPOINTS_LOCATION, "siamese_7_class.hdf5")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='max')

tensorboard = TensorBoard(log_dir=LOGS_LOCATION, histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [checkpoint, earlystop, tensorboard]


model = siamese_net
X_train_instances = [X_train_left, X_train_right]
X_test_instances = [X_test_left, X_test_right]

hist = model.fit(X_train_instances, y_train, shuffle=True, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS, verbose=1, validation_data=(X_test_instances, y_test), callbacks=callbacks_list)
