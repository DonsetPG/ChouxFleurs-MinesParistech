
################################################################

"""

Import des différents modules

"""

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

###################################################################

# On va maintenant charger notre réseau de neurones :

json_file = open('/home/pi/Desktop/model_small.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/home/pi/Desktop/model_small.h5")

model = loaded_model
print("model : ON")

#cf_model = load_model(r'/home/pi/Desktop/xception_cf_v1.model')


TOTAL_CONSEC = 0


###################################################################

# Mise en place des paramètres :

latenceStart = 2.0

runningTime = 500

WIDTH, HEIGHT = 224,224

NORMALISATION = 255.


###################################################################

# Début de l'activation de la caméra :

print("start Video Stream")

vs = VideoStream(usePiCamera = True).start()
time.sleep(latenceStart)

begin = time.time()
end = time.time()

# Début du programme de détection :

while (end - begin) < runningTime:
    end = time.time()

    frame = vs.read()
    frame = imutils.resize(frame,width=WIDTH)

    image = cv2.resize(frame,(WIDTH,HEIGHT))
    image = image.astype("float") / NORMALISATION
    image = img_to_array(image)

    image = np.expand_dims(image,axis=0)

    (mur,pasMur) = model.predict(image)[0][0:2]

    label = "pasMur"

    proba = pasMur

    if mur > pasMur:
        label = "mur"
        proba = mur
        TOTAL_CONSEC += 1


    label = "{}: {:.2f}%".format(label,proba*100)
    frame = cv2.putText(frame,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,224,0),2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

print("cleaning")
cv2.destroyAllWindows()
vs.stop()
