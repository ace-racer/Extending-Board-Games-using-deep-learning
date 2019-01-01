# Keras and TF imports
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input

# Scikit-learn and Numpy imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow import set_random_seed
set_random_seed(42)





X_train, y_train = get_required_data_with_labels_for_model(location_of_train_data)
print(len(X_train))
print(len(y_train))
print(X_train[0].shape)
print(y_train[0])

X_test, y_test = get_required_data_with_labels_for_model(location_of_test_data)
print(len(X_test))
print(len(y_test))
print(X_test[0].shape)
print(y_test[0])

## update the base inception v3 model

num_output_classes = len(class_names)
print(num_output_classes)

# create the base pre-trained model
inception_v3_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = inception_v3_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

predictions = Dense(num_output_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=inception_v3_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in inception_v3_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

if not os.path.exists(model_folder_name):
    os.makedirs(model_folder_name)

if not os.path.exists(tensorboard_logs_folder_location):
    os.makedirs(tensorboard_logs_folder_location)

# checkpoint
filepath=os.path.join(model_folder_name, "chess_pieces_inceptionv3_p1.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=20, min_delta= 0.0001)
tensorboard = TensorBoard(log_dir=tensorboard_logs_folder_location, histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint, early_stopping, tensorboard]

epochs = 100
batch_size = 100

X_train = np.array(X_train)
X_test = np.array(X_test)
history = model.fit(X_train,
          y_train,
          epochs=epochs,
          validation_data=(X_test, y_test),
          verbose=1,
          callbacks=callbacks_list,
          batch_size=batch_size)

## Fine tune some inception layers
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# checkpoint
filepath=os.path.join(model_folder_name, "chess_pieces_inceptionv3_p2.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=25, min_delta= 0.01)
callbacks_list = [checkpoint, early_stopping, tensorboard]

epochs = 100
batch_size = 100

history_2 = model.fit(X_train,
          y_train,
          epochs=epochs,
          validation_data=(X_test, y_test),
          verbose=1,
          callbacks=callbacks_list,
          batch_size=batch_size)

model.load_weights(os.path.join(model_folder_name, "chess_pieces_inceptionv3_p2.hdf5"))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=SGD(lr=0.0001, momentum=0.9),
    metrics=['accuracy'],
)
score = model.evaluate(X_test, y_test, verbose=0)

print("Score: " + str(score))

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

test_predictions = model.predict(X_test, batch_size=batch_size)
y_test_pred = [np.argmax(x) for x in test_predictions]
cnf_matrix = confusion_matrix(y_test, y_test_pred)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
