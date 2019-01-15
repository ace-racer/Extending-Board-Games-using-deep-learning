# An application to extend board games like chess to digital forms
## Use cases:
1. Chess is just an example and the application is designed such that it is easily extended for any board game
2. Players of the board games are in different geographies
3. Input for a intelligent robot to play board games
And many more....

## Source code structure
    .
    ├── API                                 # Contains code for the API described in the architecture
    │   ├── server.py                       # The python flask API server
    │   ├── requestprocessor.py             # Contains implementations to process each of the requests received by the server
    │   ├── configurations.py               # Contains the configurations (or the values that will change based on the environment)
    │   |── constants.py                    # The constants used in the API
    │   |── mongodbprovider.py              # Contains methods to interact with Mongo DB
    │   |── redisprovider.py                # Contains methods to interact with Redis
    │   |── chess_board_segmentation.py     # Contains methods to segment the chess board image into 64 sub images
    │   |── chess_piece_recognition.py      # Contains methods to infer the chess piece based on the trained deep learning model
    │   |── mock_server.py                  # Implement a mock server to help initial integration with client app
    │   |── utils.py                        # Contains utility methods used across files
    │   |── tests.py                        # Contains tests to check different scenarios
    ├── chess_board_segmentation            # Contains notebooks to evaluate different algorithms to segment chess board
    |── chess_piece_detection               # Contains code for different deep learning models trained to detect chess pieces
    │   ├── app
    |       ├── appconfigs.py               # Different configurations that are required for training the models (like location of training  images etc.)
    │       ├── modelconfigs.py             # Contains the configurations for the different models used during training the model
    │       |── constants.py                # The constants used across the training app
    │       ├── models.py                   # Contains the different models that are trained
    │       ├── data_generator.py           # Contains the data generator for generating the training data in batches. [Refer]                 (https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)      
    │       ├── utils.py                    # Contains the utility methods for obtaining the data, printing confusion matrix etc.
    │       └── app.py                      # Tie all codes together and train or evaluate the required model based on the arguments passed
    |── training_data_analysis              # Contains code to analyze the training data
    └── training_data_generation            # Contains code to generate and crawl for training chess piece images


## System architecture
The system is designed as a REST based application and a separate client consumes the different services that are exposed.
![Architecture](/doc/arch_v1.png) 

## Image recognition modules

1. Transfer learning perfomred on InceptionV3 pretrained on ImageNet to recognize chess piece images. Refer [here](https://keras.io/applications/#inceptionv3) for details.
Details on inception modules [here](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/)

2. Custom CNN implementation to recognize chess piece images

## References
1. https://medium.com/@daylenyang/building-chess-id-99afa57326cd
2. https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/