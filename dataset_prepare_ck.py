# Ce fichier n'appartinent pas au code principal

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pixeltomatrix import SetPixelInMatrix
from makedirectories import makeDir
from matplotlib import pyplot as histo

size = 48  # image size = 48x48
makeDir()  # Création des dossiers

# Initializing a counter for each subcategory in both "train" and "test" -> for testing
angry_train = 0
disgusted_train = 0
fearful_train = 0
happy_train = 0
sad_train = 0
surprised_train = 0
neutral_train = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

# Ouvrerture du fichier du Dataset et et initialisation d'une matrice
ds = pd.read_csv("ckextended.csv")
mat = np.zeros((size, size), dtype=np.uint8)
print("Enregistrement des images... Patientez svp")

# Lire le fichier csv ligne par ligne (i=image, j= pixel)
for i in tqdm(range(len(ds))):
    txt = ds["pixels"][i]
    list = txt.split()

    for j in range(size * size):
        SetPixelInMatrix(j, mat, list, size)

    img = Image.fromarray(mat)  # On construit chaque image à partir de sa matrice
    # (the 28709 first images belong in the train folder)
    # Enregistrement des images dans le dossier train, suivant l'émotion correspondante
    if i < 733:
        if ds["emotion"][i] == 0:
            img.save("datack/train/angry/im" + str(angry_train) + ".png")
            angry_train += 1
        elif ds["emotion"][i] == 1:
            img.save("datack/train/disgusted/im" + str(disgusted_train) + ".png")
            disgusted_train += 1
        elif ds["emotion"][i] == 2:
            img.save("datack/train/fearful/im" + str(fearful_train) + ".png")
            fearful_train += 1
        elif ds["emotion"][i] == 3:
            img.save("datack/train/happy/im" + str(happy_train) + ".png")
            happy_train += 1
        elif ds["emotion"][i] == 4:
            img.save("datack/train/sad/im" + str(sad_train) + ".png")
            sad_train += 1
        elif ds["emotion"][i] == 5:
            img.save("datack/train/surprised/im" + str(surprised_train) + ".png")
            surprised_train += 1
        elif ds["emotion"][i] == 6:
            img.save("datack/train/neutral/im" + str(neutral_train) + ".png")
            neutral_train += 1

    # Enregistrement des images dans le dossier test, suivant l'émotion correspondante
    else:
        if ds["emotion"][i] == 0:
            img.save("datack/test/angry/im" + str(angry_test) + ".png")
            angry_test += 1
        elif ds["emotion"][i] == 1:
            img.save("datack/test/disgusted/im" + str(disgusted_test) + ".png")
            disgusted_test += 1
        elif ds["emotion"][i] == 2:
            img.save("datack/test/fearful/im" + str(fearful_test) + ".png")
            fearful_test += 1
        elif ds["emotion"][i] == 3:
            img.save("datack/test/happy/im" + str(happy_test) + ".png")
            happy_test += 1
        elif ds["emotion"][i] == 4:
            img.save("datack/test/sad/im" + str(sad_test) + ".png")
            sad_test += 1
        elif ds["emotion"][i] == 5:
            img.save("datack/test/surprised/im" + str(surprised_test) + ".png")
            surprised_test += 1
        elif ds["emotion"][i] == 6:
            img.save("datack/test/neutral/im" + str(neutral_test) + ".png")
            neutral_test += 1


x = np.array(['Colère','Dégoût','Peur','Joie','Neutralité','Tristesse','Surprise'])
histo.title("Distributuon du Dataset ck+")
fig = histo.bar(x,[angry_train, disgusted_train, fearful_train, happy_train, neutral_train, sad_train, surprised_train]); #Plot prédiction x=x et y=prediction[0]
histo.savefig("ck+_distribution.png")
histo.show()
print("Angry : ", angry_train, " images")
print("Disgusted : ", disgusted_train, " images")
print("Fearful : ", fearful_train, " images")
print("Happy : ", happy_train, " images")
print("Neutral : ", neutral_train, " images")
print("Sad : ", sad_train, " images")
print("Surprised : ", surprised_train, " images")
print("Nombre de données d'entrainenment : ", angry_train + disgusted_train + fearful_train + happy_train + neutral_train + sad_train + surprised_train, " images")
print("Nombre de données de test : ", angry_test + disgusted_test + fearful_test + happy_test + neutral_test + sad_test + surprised_test, " images")
print("Terminé avec succés!")

