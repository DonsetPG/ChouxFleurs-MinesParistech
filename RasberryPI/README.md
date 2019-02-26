# Matériel : 

Il faudra une rasberry PI qui fonctionne, ainsi qu'une caméra compatible rasberry PI. Nous avons ici utilisé [cette caméra](https://www.amazon.fr/Raspberry-Pi-1080p-Module-Caméra/dp/B01ER2SKFS/ref=sr_1_3?ie=UTF8&qid=1551220051&sr=8-3&keywords=camera+raspberry+pi).

Il faudra de plus autoriser la caméra à fonctionner sur la rasberry PI utilisée, ouvrir le terminal et taper : 

```
sudo raspi-config
```

Une fois dans le menu, rendez-vous dans **Interfacing Options** puis dans **caméra**. Il faut ensuite redemarrer la rasberry PI.

# Librairies : 

Le réseau de neurones a été entrainé avec Keras et Tensorflow. Nous allons donc devoir ici utiliser Keras pour charger notre modèle.
La structure du modèle et les différents poids sont présents dans le fichier * *.zip* * modelDetection. Placer le dans le même emplacement
que le script * *cf_detector.py* *.  

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

A ce stade la, tout peut maintenant fonctionner, il suffit de se placer dans la bonne direction dans un terminal, et de taper : 

```
python3 cf_detector.py
```

Une fenêtre avec la vision de la caméra va alors s'ouvrir, et la probabilité de détection devrait alors s'afficher.

On va cependant expliquer les différentes parties du code ici. 

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




