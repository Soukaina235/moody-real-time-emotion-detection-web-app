from keras.optimizers import Adam
from plotmodelhistory import plot_model_history


def train(
    model,
    train_generator,
    num_train,
    batch_size,
    num_epoch,
    validation_generator,
    num_val,
):
    model.compile(
        loss="categorical_crossentropy", # specifies the loss function to optimize during training. Here=> categorical cross-entropy(suitable for for multi-class classification problems)
        optimizer=Adam(learning_rate=0.0001, decay=1e-6), # Adam dynamically adjusts the model's parameters based on the negative gradients of the loss function
        metrics=["accuracy"], # une métrique contenant la précision précedente et la précision actuelle
    )
    model_info = model.fit( # model_info : l'historique du modèle
        train_generator, 
        steps_per_epoch=num_train // batch_size, # étapes par époche pour entrainement(nb de batchs par epoch)
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size, # étapes par époche pour test(nb de batchs par epoch)
    )

    plot_model_history(model_info) # trace les courbes d'exactitude et de perte 
    model.save_weights("model.h5") # Enregistre les poids (les paramètres) du modèle   # changez le chemin pour les autres modèles

    ck_json = model.to_json()  # Convertit le modèle au format to_json()
    with open("fer.json", "w") as json_file: # Ecrit la représentation JSON du model dans fer.json  # changez le chemin pour les autres modèles
        json_file.write(ck_json)

