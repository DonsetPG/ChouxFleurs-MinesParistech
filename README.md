# ChouxFleurs-MinesParistech

# Repérage des choux-fleurs mûrs: 

- https://www.ijabe.org/index.php/ijabe/issue/view/70 : article "High performance vegetable classification from images based on AlexNet deep learning model" par Ling Zhu,	Zhenbo Li,	Chen Li,	Jing Wu et Jun Yue. En s'inspirant de cet article, nous allons pouvoir faire deux choses qui vont grandement nous aider : 
- 1. utiliser des images en plus de la base de données ImagesNet (https://fr.wikipedia.org/wiki/ImageNet). 
- 2. Réutiliser les premières couches entrainées(!) du réseaux de neuronnes AlexNet (https://en.wikipedia.org/wiki/AlexNet) et en les gelant lors de l'apprentissage de notre dataset. Via une utilisation simple d'AlexNet, le papier précédent obtient une précision de 92.1% pour 5 catégories de légumes : "broccoli, pumpkin, cauliflower, mushrooms and cucumber". En réutilisant les premières couches, en réduisant le nombre de catégories à deux, ainsi qu'en utilisant en plus notre dataset réalisé dans des exploitations de choux-fleurs, les résultats atteignables sont excellents. 

- L'implémentation réalisée du réseaux de neurones se trouve dans le dossier NN-classification sous le nom cauliflower-alexnet-v1.py.
- Les paramètres utilisés se trouvent dans cauliflower-nn-args.py.
- l'architecture du réseaux peut être trouvée https://arxiv.org/abs/1608.06993 page 4.
- On préférera même utiliser un réseaux DenseNet (DCCN), dont les performances sont maintenant meilleures qu'AlexNet. 
 
- Les différentes images et vidéos se trouvent dans le fichier data du drive. Pour augmenter la taille du dataset (ensemble des images), on va pouvoir utiliser les différentes fonctions suivantes : 

```python
import tensorflow as tf 
image = tf.image.flip_left_right(image)
image = tf.image.flip_up_down(image)
image = tf.image.flip_up_down(tf.image.flip_left_right(image))
image = tf.image.random_brightness(image,alpha)
image = tf.image.random_contrast(image,minContrast,maxContrast)
image = tf.image.random_jpeg_quality(image,min,max)
image = tf.image.random_saturation(image,minSat,maxSat)
``` 

- Résumé du modèle utilisé : 
``` 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________

conv2d_1 (Conv2D)            (None, 54, 54, 96)        34944     
_________________________________________________________________
activation_1 (Activation)    (None, 54, 54, 96)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 27, 27, 96)        384       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 23, 23, 256)       614656    
_________________________________________________________________
activation_2 (Activation)    (None, 23, 23, 256)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 11, 11, 256)       0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 11, 11, 256)       1024      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 9, 384)         885120    
_________________________________________________________________
activation_3 (Activation)    (None, 9, 9, 384)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 9, 9, 384)         1536      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 7, 384)         1327488   
_________________________________________________________________
activation_4 (Activation)    (None, 7, 7, 384)         0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 7, 7, 384)         1536      
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 5, 5, 256)         884992    
_________________________________________________________________
activation_5 (Activation)    (None, 5, 5, 256)         0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 256)         0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 2, 2, 256)         1024      
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              4198400   
_________________________________________________________________
activation_6 (Activation)    (None, 4096)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 4096)              16384     
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
activation_7 (Activation)    (None, 4096)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 4096)              0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 4096)              16384     
_________________________________________________________________
dense_3 (Dense)              (None, 1000)              4097000   
_________________________________________________________________
activation_8 (Activation)    (None, 1000)              0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 1000)              0         
_________________________________________________________________
batch_normalization_8 (Batch (None, 1000)              4000      
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 2002      
_________________________________________________________________
activation_9 (Activation)    (None, 2)                 0     

Total params: 28,868,186
Trainable params: 28,847,050
Non-trainable params: 21,136
``` 
