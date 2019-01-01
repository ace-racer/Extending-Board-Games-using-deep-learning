# Scikit-learn and Numpy imports
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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
    
    # if the first argument is True ish
    if sys.argv[1]:
        history, model = models.train_InceptionV3_transfer_learning_model(X_train, y_train, X_test, y_test, inceptionV3configs)
        
        # if the history needs to be plotted
        if appconfigs.plot_history:
            utils.plot_train_validation_accuracy(history)
        
        utils.get_score_confusion_matrix(X_test, y_test, model, inceptionV3configs, False)
    else:
        _, model = models.train_InceptionV3_transfer_learning_model(X_train, y_train, X_test, y_test, inceptionV3configs, False)
        utils.get_score_confusion_matrix(X_test, y_test, model, inceptionV3configs)
    