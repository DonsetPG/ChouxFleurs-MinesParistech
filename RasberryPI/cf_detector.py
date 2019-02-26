
from keras.preprocessing.image import img_to_array
import numpy as np
import keras
from keras.models import Model, load_model

from keras.models import model_from_json
import os

print("done")

json_file = open('/home/pi/Desktop/model_small.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/home/pi/Desktop/model_small.h5")

model = loaded_model
print("model : ON")
#cf_model = load_model(r'/home/pi/Desktop/xception_cf_v1.model')

from imutils.video import VideoStream
from threading import Thread
import cv2
import imutils
import time, os
TOTAL_CONSEC = 0
#################

print("start Video Stream")

vs = VideoStream(usePiCamera = True).start()
time.sleep(2.0)

begin = time.time()
end = time.time()

while (end - begin) < 500:
    frame = vs.read()
    
    frame = imutils.resize(frame,width=224)
    end = time.time()

    image = cv2.resize(frame,(224,224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)

    (mur,pasMur) = model.predict(image)[0]
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





    
