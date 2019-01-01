# Scikit-learn and Numpy imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# TF/Keras imports
from tensorflow import set_random_seed

# custom imports
import utils
import appconfigs
import constants
import models
import modelconfigs

set_random_seed(42)

if __name__ == "__main__":
    # create the required folders
    utils.create_artifact_folders()

    X_train, y_train = utils.get_required_data_with_labels_for_model(appconfigs.location_of_train_data)
    print(len(X_train))
    print(len(y_train))
    print(X_train[0].shape)
    print(y_train[0])

    X_test, y_test = utils.get_required_data_with_labels_for_model(appconfigs.location_of_test_data)
    print(len(X_test))
    print(len(y_test))
    print(X_test[0].shape)
    print(y_test[0])

    inceptionV3configs = modelconfigs.inceptionV3configs
    history, model = models.train_InceptionV3_transfer_learning_model(X_train, y_train, X_test, y_test, inceptionV3configs)
