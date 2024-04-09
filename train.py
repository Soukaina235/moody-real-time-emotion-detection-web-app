from keras.preprocessing.image import ImageDataGenerator
import os

from TrainingFunction import train
from model import MakeModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Training data
train_dir = "data/train"
# Validation data
val_dir = "data/test"

# Pour entraîner le modèle sur ck+: changer les lignes 10 et 12 par
# train_dir = "datack/train"
# val_dir = "datack/test"

# The number of images in each set
num_train = 28709 # Pou ck+ : changer cette valeur par 733
num_val =  7178 # Pou ck+ : changer cette valeur par 186
batch_size = 64 # The number of images processed at once
num_epoch = 50 #  the number of times the model will be trained on the dataset

# Preprocessing
# Normalisation
train_datagen = ImageDataGenerator(rescale=1.0 / 255) #crée une instance de ImageDataGenerator pour training data. / rescale => normalisation
val_datagen = ImageDataGenerator(rescale=1.0 / 255) #crée une instance de ImageDataGenerator pour testing data. / rescale => normalisation

train_generator = train_datagen.flow_from_directory( # La méthode flow_from_directory() génère des batches de données augumentées pour l'entrainement
    train_dir,  # Path to training imgs
    target_size=(48, 48),  # img size :the desired size to which all images will be resized during the preprocessing step
    batch_size=batch_size,
    color_mode="grayscale", # mode de couleur:"grayscale"
    class_mode="categorical", # spécifie le type des class labels que le générateur doit retourner => ici ça retourne multi-class classification labels
)

# Same
validation_generator = val_datagen.flow_from_directory(
    val_dir, 
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
)

# Create the model
model = MakeModel()
train(
    model,
    train_generator,
    num_train,
    batch_size,
    num_epoch,
    validation_generator,
    num_val,
)



