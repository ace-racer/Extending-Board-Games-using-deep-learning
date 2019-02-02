"""
Initial code from: https://sorenbouma.github.io/blog/oneshot/

"""
import keras
from keras.layers import Input, Conv2D, Lambda, average, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (100, 100, 3)
left_input = Input(input_shape)
right_input = Input(input_shape)

#build convnet to use in each siamese 'leg'
convnet = Sequential()

convnet.add(Conv2D(32,(5,5),input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
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
both = average([encoded_l,encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
#optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

optimizer = Adam(0.00006)

siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

print(siamese_net.count_params())
print(siamese_net.summary())