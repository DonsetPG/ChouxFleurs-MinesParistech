import tensorflow as tf
import numpy as np
import os
import copy
from labels import get_labels
import numpy as np 


NB_IMG = 127

## Location of the pictures and videos : 

filepath = '/Users/paulgarnier/Desktop/Files/ChouxFleurs/frames/'
ext = '.jpg'

## Getting labels : 

indx_img = get_labels(False)
labels_img = []
for i in range(1,NB_IMG):
    if i in indx_img:
        labels_img.append([1,0])
    else:
        labels_img.append([0,1])

## Getting the img names : 

name_img = []

for i in range(1,NB_IMG):
    name = filepath+'im'+str(i)+ext
    name_img.append(name)

# Getting them into tf : 

name_img = tf.constant(name_img)
labels_img = tf.constant(labels_img)

# Creation of a Dataset : 

dataset = tf.data.Dataset.from_tensor_slices((name_img, labels_img))

# Creation of map-function to rescale our pictures, and change them a bit : 

def _rescale_process(filename, label,shapeCheking = False,flipLR = False,flipUD = False,flipUDLR = False,randomB = False):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, 
                                            channels=3) # Channels 3 := keep them in RGB, not 
                                                        # grayscale 
    # Normalize them 
    image = tf.cast(image_decoded, tf.float32)*(1.0/255.0)  - .5
    # Rescale them to 299x299, in order to use Xception after that 
    image = tf.image.resize_images(image, [299, 299])
    if shapeCheking:
        print("shape of img : ",image.shape," ",image[0])
    if flipLR:
        image = tf.image.flip_left_right(image)
    if flipUD:
        image = tf.image.flip_up_down(image)
    if flipUDLR:
        tf.image.flip_up_down(tf.image.flip_left_right(image))
    if randomB:
        image = tf.image.random_brightness(image,0.09)
    return image, label

# We can now map our dataset : 

dataset = dataset.map(_rescale_process)

# and we create an iterator to convert it into a npy.array : 

iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

# We then compute it into a .npy file and a numpy array : 

y_train = np.array(labels_img)
x_train = []
with tf.Session() as sess:
    for _ in range(NB_IMG):
        array_classique = images.eval(session=sess)

        x_train.append(array_classique)

x_train = np.array(x_train)

np.save('/Users/paulgarnier/Desktop/Files/cauliflower_x',x_train)
np.save('/Users/paulgarnier/Desktop/Files/cauliflower_y',y_train)

def get_dataset():
    return x_train,y_train


