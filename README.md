# ChouxFleurs-MinesParistech

# Repérage des choux-fleurs mûrs: 

- https://www.ijabe.org/index.php/ijabe/issue/view/70 : article "High performance vegetable classification from images based on AlexNet deep learning model" par Ling Zhu,	Zhenbo Li,	Chen Li,	Jing Wu et Jun Yue. En s'inspirant de cet article, nous allons pouvoir faire deux choses qui vont grandement nous aider : 
- 1. utiliser des images en plus de la base de données ImagesNet (https://fr.wikipedia.org/wiki/ImageNet). 
- 2. Réutiliser les premières couches entrainées(!) du réseaux de neuronnes AlexNet (https://en.wikipedia.org/wiki/AlexNet) et en les gelant lors de l'apprentissage de notre dataset. Via une utilisation simple d'AlexNet, le papier précédent obtient une précision de 92.1% pour 5 catégories de légumes : "broccoli, pumpkin, cauliflower, mushrooms and cucumber". En réutilisant les premières couches, en réduisant le nombre de catégories à deux, ainsi qu'en utilisant en plus notre dataset réalisé dans des exploitations de choux-fleurs, les résultats atteignables sont excellents. 

- L'implémentation réalisée du réseaux de neurones se trouve dans le dossier NN-classification sous le nom cauliflower-alexnet-v1.py.
- Les paramètres utilisés se trouvent dans cauliflower-nn-args.py.
- l'architecture du réseaux peut être trouvée https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf page 5.

- Les différentes images et vidéos se trouvent dans le fichier data. Pour augmenter la taille du dataset (ensemble des images), on va pouvoir utiliser les différentes fonctions suivantes : 

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

