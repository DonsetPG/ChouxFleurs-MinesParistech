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
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

#from data_processing import get_dataset 
from set_up_transfer_learning import setup_to_transfer_learn,add_new_last_layer,freeze_some

###################


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 25
BAT_SIZE = 32
FC_SIZE = 1024
NB_IMG_TRAIN = 246 + 404
NB_IMG_VAL = 15+63

##################
print("loading data....")
#x_train,y_train = get_dataset()
print("data loaded")


#################

def train(args):
    
    train_img = '/Users/paulgarnier/Desktop/Files/ChouxFleurs/frames/' 
    validation_img = '/Users/paulgarnier/Desktop/Files/ChouxFleurs/validation/' 
    
    nb_epoch = int(args.nb_epoch)
    #nb_train_samples = get_nb_files(train_img)
    nb_classes = 2
    # data prep
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.01,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.01,
        height_shift_range=0.01,
        shear_range=0.01,
        zoom_range=0.01,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest')

    
    train_generator = train_datagen.flow_from_directory(
			train_img,
			target_size=(299, 299),
			batch_size=BAT_SIZE,
            color_mode='rgb',
			class_mode='categorical'
			)
    validation_generator = validation_datagen.flow_from_directory(
			validation_img,
			target_size=(299, 299),
			batch_size=BAT_SIZE,
            color_mode='rgb',
			class_mode='categorical'
			)
    """
    if(K.image_dim_ordering() == 'th'):
        input_tensor = Input(shape=(3, 299, 299))
    else:
        input_tensor = Input(shape=(299, 299, 3))
    """
    input_tensor = Input(shape=(299, 299, 3))
    # setup model
    base_model = Xception(input_tensor = input_tensor,weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)
    #model = load_model('/Users/paulgarnier/Desktop/Files/GitHub/cauliflower/inceptionv3-ft.model')
    # transfer learning
    #setup_to_transfer_learn(model, base_model)
    freeze_some(model,base_model,125)
    
    history_tl = model.fit_generator(train_generator,steps_per_epoch=NB_IMG_TRAIN/BAT_SIZE,epochs=nb_epoch,validation_data=validation_generator,validation_steps=NB_IMG_VAL/BAT_SIZE) 
    #history_tl = model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=BAT_SIZE),steps_per_epoch=len(x_train) /BAT_SIZE,epochs=nb_epoch)

    model.save(args.output_model_file)
    
    plot_training(history_tl)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'b+',label='Training accuracy')
    plt.plot(epochs, val_acc, 'g--',label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png')
    plt.show()
    plt.figure()
    plt.plot(epochs, loss, 'r+',label='Training loss')
    plt.plot(epochs, val_loss, 'b--',label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig('loss.png')
    plt.show()

