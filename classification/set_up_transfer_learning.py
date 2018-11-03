import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras import backend as K
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

###################


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 10
BAT_SIZE = 32
FC_SIZE = 1024
NB_IMG = 127

##################

def setup_to_transfer_learn(model, base_model):
    # We freeze every layer for transfer learning : 
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
        
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = AveragePooling2D((8, 8), border_mode='valid', name='avg_pool')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


