# An application to extend board games like chess to digital forms
## Use cases:
1. Chess is just an example and the application is designed such that it is easily extended for any board game
2. Players of the board games are in different geographies
3. Input for a intelligent robot to play board games
And many more....

## Source code structure
TODO

## System architecture
The system is designed as a REST based application and a separate client consumes the different services that are exposed.
![Architecture](/doc/arch_v1.png) 

## Image recognition modules

1. Transfer learning perfomred on InceptionV3 pretrained on ImageNet to recognize chess piece images. Refer [here](https://keras.io/applications/#inceptionv3) for details.

2. Custom CNN implementation to recognize chess piece images

## References
1. https://medium.com/@daylenyang/building-chess-id-99afa57326cd
2. https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/