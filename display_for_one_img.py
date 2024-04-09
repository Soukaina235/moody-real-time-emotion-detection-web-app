import numpy as np
import cv2
import os
from keras.utils import img_to_array
from matplotlib import pyplot as histo

from model import MakeModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = MakeModel()

#Charger le model
model.load_weights('model.h5')

# Dictionnaire contenant les émotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

impath = r'imgs_for_display/imagetest3.jpg' # Le lien vers l'image
frame = cv2.imread(impath) # Lire l'image
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Rendre l'image en gris
# Détecte les visages dans l'image en utilisant le classificateur de cascade Haar.
faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces: # Itérer dans les visages détectés: x,y:coords, w=width, h=height
    cv2.rectangle(frame, (x, y), (x + w, y + h), (128,0,128), thickness=2) #Dessiner un rectangle autour du visage
    roi_gray = gray[ # roi_gray = région d'intéret = visage
        y : y + w, x : x + h
    ]  
    roi_gray = cv2.resize(roi_gray, (48, 48)) # Redimmensionner roi
    img_pixels = img_to_array(roi_gray) # Convertir l'image en pixels
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255 # Normalisation
    predictions = model.predict(img_pixels) # Prédiction matrice 1x1 avec chaque ligne et une liste ie. [[]]
    maxindex = int(np.argmax(predictions)) # Récuperer l'indice de l'émotion prédite

print(emotion_dict[maxindex]) # Afficher l'émotion prédite
# print(predictions)

x = np.array(['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised'])

histo.bar(x,predictions[0]) #Plot prédiction x=x et y=prediction[0]
histo.savefig("imgs_for_display/result.png")
histo.show() # Afficher histogramme

