from flask import Flask, render_template, Response, send_file #  Importe les classes et fonctions nécessaires de Flask pour créer une application web, rendre des modèles, gérer les réponses et envoyer des fichiers
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.utils import img_to_array
import csv

# charger le model
model = model_from_json(open("fer.json", "r").read()) # on utilise notre modèle principal

# associer les poids enregistrés avec le modèle précédemment chargé
model.load_weights("model.h5") # on utilise notre modèle principal

# Charger haar
face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

app = Flask(__name__) # Créer une instance de l'app Flask

camera = cv2.VideoCapture(0) #  Initialise la capture vidéo en utilisant la caméra par défaut (index 0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Rendre l'image en gris

            # Détecte les visages dans l'image en utilisant le classificateur de cascade Haar.
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for x, y, w, h in faces_detected: # Itérer dans les visages détectés: x,y:coords, w=width, h=height
                print("WORKING")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128,0,128), thickness=2) #Dessiner un rectangle autour du visage
                roi_gray = gray_img[
                    y : y + w, x : x + h
                ]  # roi_gray = région d'intéret = visage
                roi_gray = cv2.resize(roi_gray, (48, 48)) # Redimmensionner roi 
                img_pixels = img_to_array(roi_gray) # Convertir l'image en pixels
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255 # Normalisation

                predictions = model.predict(img_pixels) # Prédiction matrice 1x1 avec chaque ligne et une liste ie. [[]]
                max_index = np.argmax(predictions[0]) # Récuperer l'indice de l'émotion prédite

                emotions = [
                    "Angry",
                    "Disgusted",
                    "Fearful",
                    "Happy",
                    "Neutral",
                    "Sad",
                    "Suprised",
                ]
                
                predicted_emotion = emotions[max_index]

                # Liste qui contient l'émotion prédite
                row_list = [[predicted_emotion]]
                # Ouvrir prediction.cvs dans le mode ajout
                with open('predictions.csv', 'a', newline='') as file:
                    writer = csv.writer(file) # Ecrire l'émotion dans une nouvelle ligne
                    writer.writerows(row_list)
                
                print(predicted_emotion)
                cv2.putText( # Afficer l'émotion prédite
                    frame, 
                    predicted_emotion,
                    (int(x), int(y)), # Position du départ du texte
                    cv2.FONT_HERSHEY_SIMPLEX, # La police
                    1, # La taille de police
                    (255, 255, 255), # Couleur du texte
                    2, # Epaisseur du texte
                )

            # resized_img = cv2.resize(frame, (1000, 700))
            ret, buffer = cv2.imencode(".jpg", frame) # L'image est encodée du format binaire => fomat JPEG

            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )  # concat frame one by one and show result


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame") 

@app.route("/")
def index():
    return render_template("index.html") #generate output from a template file based on the Jinja2 engine that is found in the application's templates folder


@app.route('/templates/about.html')
def about():
    return render_template('about.html')

@app.route('/templates/Model.html')
def Model():
    return render_template('Model.html')


@app.route('/download')
def download():
    path = 'predictions.csv'
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
