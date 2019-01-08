# Scikit-learn and Numpy imports
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.environ['KERAS_BACKEND'] = 'tensorflow'

# TF/Keras imports
from tensorflow import set_random_seed

# custom imports
import utils
import appconfigs
import constants
import models
import modelconfigs
from other_models import alexnet

set_random_seed(42)

"""
    Usage: python app.py 0 "inception"
"""
if __name__ == "__main__":
    # create the required folders
    utils.create_artifact_folders()
    model_name = sys.argv[2]
    
    configs_to_use = None
    model_trainer = None

    if model_name == "inception":
        configs_to_use = modelconfigs.inceptionV3configs
        model_trainer = models.train_InceptionV3_transfer_learning_model
    elif model_name == "cnn":
        configs_to_use = modelconfigs.customCNNconfigs
        model_trainer = models.train_custom_cnn_model
    elif model_name == "alexnet":
        configs_to_use = modelconfigs.alexNetconfigs
        model_trainer = alexnet.train_alexnet_model
    # TODO: add other models
    else:
        raise ValueError("Model name not mentioned correctly")
    
    # if the first argument is 1
    if sys.argv[1] == "1":
        print("Training model...")
        history, model, X_test, y_test = model_trainer(configs_to_use, True)
        
        # if the history needs to be plotted
        if appconfigs.plot_history:
            utils.plot_train_validation_accuracy(history)
        
        utils.get_score_confusion_matrix(X_test, y_test, model, configs_to_use, False)
    else:
        _, model, X_test, y_test = model_trainer(configs_to_use, False)
        utils.get_score_confusion_matrix(X_test, y_test, model, configs_to_use)
    