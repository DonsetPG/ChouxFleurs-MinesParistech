import keras

# Importation des différentes outils à utiliser;
# On réalise ici une implementation à l'aide de Keras;

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from cauliflower-nn-args import get_args

# On importe les arguments du réseaux :

args = get_args()

SIZE_IMAGE = args.size_image
N_CLASSES = args.n_classes
ACTIVATION_OUTPUT = args.activation_output
DROPOUT_VALUE = args.dropout_value
ACTIVATION_INSIDE = args.activation_inside

# Début de la construction du modèle :

nn_alexnet = Sequential()

# Première couche : Convolution qui filtre avec 96 kernels de taille 11 x 11 et un stride de 4 pixels.

nn_alexnet.add(Conv2D(filters=96, input_shape=SIZE_IMAGE, kernel_size=(11,11),strides=(4,4), padding='valid'))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
nn_alexnet.add(BatchNormalization())

# Deuxième couche : Convolution qui filtre avec 256 kernels de taille 5 x 5.
nn_alexnet.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
nn_alexnet.add(BatchNormalization())

# Troisième couche : Convolution qui filtre avec 384 kernels de taille 3 x 3.
nn_alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(BatchNormalization())

# Quatrième couche : Convolution qui filtre avec 384 kernels de taille 3 x 3.
nn_alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(BatchNormalization())

# Cinquième couche : Convolution qui filtre avec 256 kernels de taille 3 x 3.
nn_alexnet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
nn_alexnet.add(BatchNormalization())

# Couche dense :
nn_alexnet.add(Flatten())
# Première :
nn_alexnet.add(Dense(4096, input_shape=(224*224*3,)))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(Dropout(DROPOUT_VALUE))
nn_alexnet.add(BatchNormalization())

# Seconde
nn_alexnet.add(Dense(4096))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(Dropout(DROPOUT_VALUE))
nn_alexnet.add(BatchNormalization())

# Troisième
nn_alexnet.add(Dense(1000))
nn_alexnet.add(Activation(ACTIVATION_INSIDE))
nn_alexnet.add(Dropout(DROPOUT_VALUE))
nn_alexnet.add(BatchNormalization())

# Couche de sortir
nn_alexnet.add(Dense(N_CLASSES))
nn_alexnet.add(Activation(ACTIVATION_OUTPUT))

# Résumé du modèle :
nn_alexnet.summary()

# Compilation du modèle :
nn_alexnet.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
