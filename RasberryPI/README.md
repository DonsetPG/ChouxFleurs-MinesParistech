# Matériel : 

Il faudra une rasberry PI qui fonctionne, ainsi qu'une caméra compatible rasberry PI. Nous avons ici utilisé [cette caméra](https://www.amazon.fr/Raspberry-Pi-1080p-Module-Caméra/dp/B01ER2SKFS/ref=sr_1_3?ie=UTF8&qid=1551220051&sr=8-3&keywords=camera+raspberry+pi).

Il faudra de plus autoriser la caméra à fonctionner sur la rasberry PI utilisée, ouvrir le terminal et taper : 

```
sudo raspi-config
```

Une fois dans le menu, rendez-vous dans **Interfacing Options** puis dans **caméra**. Il faut ensuite redemarrer la rasberry PI.

# Librairies : 

Le réseau de neurones a été entrainé avec Keras et Tensorflow. Nous allons donc devoir ici utiliser Keras pour charger notre modèle.
La structure du modèle et les différents poids sont présents dans le fichier  *.zip*  modelDetection. Placer le dans le même emplacement
que le script *cf_detector.py* .  

Il va maintenant falloir configurer la rasberry PI pour fonctionner avec différents modules : numpy, keras et openCV notamment. 

On trouvera ici les différentes instructions pour s'en occuper : 

###### Python, Numpy, etc : 

```
sudo apt install libatlas-base-dev
```

###### Tensorflow et Keras : 

```
pip3 install tensorflow
pip3 install keras
```

###### OpenCV : 

[Installation ici](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/)

###### Le reste : 

```
pip3 install imutils
pip3 install numpy
```

(numpy est normalement déjà installé avec la première ligne de commande)


# Le code : 

## Run : 

A ce stade la, tout peut maintenant fonctionner, il suffit de se placer dans la bonne direction dans un terminal, et de taper : 

```
python3 cf_detector.py
```

Une fenêtre avec la vision de la caméra va alors s'ouvrir, et la probabilité de détection devrait alors s'afficher.

On va cependant expliquer les différentes parties du code ici. 

## Explications : 

```python

import keras
import numpy as np
import os 
import cv2
import imutils
import time

from keras.preprocessing.image import img_to_array
from keras.models import Model, load_model
from keras.models import model_from_json

from imutils.video import VideoStream
from threading import Thread

```

On commence par importer les différents modules que l'on va utiliser.

On va ensuite mettre en place notre réseau : 

```python

json_file = open('/home/pi/Desktop/model_small.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/home/pi/Desktop/model_small.h5")

model = loaded_model

```

On commence par ouvrir le fichier * *.json* * qui contient l'architecture du modèle. Ainsi, à partir de la ligne 

```python
loaded_model = model_from_json(loaded_model_json)
```

on posède un modèle avec la bonne architecture, mais sans les poids entrainés. On va donc les charger avec la ligne suivante.
On obtient finalement notre modèle entrainé.

On va ensuite mettre en place les paramètres utilisés : 

```python

latenceStart = 2.0 # Sleeping time pour que la caméra se mette en route (en secondes)

runningTime = 500 # Temps d'activité de la caméra (en secondes)

WIDTH, HEIGHT = 224,224 # Dimensions du flux d'image de la caméra

NORMALISATION = 255. # Constante pour normaliser les pixels 

```

On va ensuite pouvoir faire commencer le flux vidéo, avec les deux lignes qui suivent : 

```python

vs = VideoStream(usePiCamera = True).start()
time.sleep(latenceStart)

```

On met ensuite en place le programme qui va tourner pendant les runningTime secondes, dans un boucle while.

```python

frame = vs.read()
frame = imutils.resize(frame,width=WIDTH)

image = cv2.resize(frame,(WIDTH,HEIGHT))
image = image.astype("float") / NORMALISATION
image = img_to_array(image)

```

Les deux premières lignes vont récupérer la frame en cours, et la mettre sous la bonne taille. 
Les trois lignes d'après normalisent l'image, la mette sous un format compatible avec notre réseau, et transforment l'image en un tableau numpy. 

```python

(mur,pasMur) = model.predict(image)[0][0:2]

```

On réalise enfin notre prédiction : existe-il un choux fleur mur sur l'image selectionnée? On se restreint ici à seulement deux classes : mur ou pasMur. 

```python

label = "pasMur"

proba = pasMur

if mur > pasMur:
  label = "mur"
  proba = mur

```

On place ici les variables label et proba sur ce que l'on a detecté (mur ou non), et la probabilité liée à cette détection. 

Finalement, on renvoit la frame selectionnée, en écrivant dessus le label selectionné ainsi que la probabilité liée : 

```python

label = "{}: {:.2f}%".format(label,proba*100)
frame = cv2.putText(frame,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,224,0),2)

cv2.imshow("Frame",frame)

```

