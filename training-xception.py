import os
import copy
import sys
import glob
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

from data_processing import get_dataset 
from set_up_transfer_learning import setup_to_transfer_learn,add_new_last_layer

from xception import train 

###################


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 10
BAT_SIZE = 32
FC_SIZE = 1024
NB_IMG = 127

#################

if __name__=="__main__":
    
    
    a = argparse.ArgumentParser()
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--plot", action="store_true")
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    args = a.parse_args()
    train(args)