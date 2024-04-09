import numpy as np
import matplotlib.pyplot as plt


def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5)) # créer une figure ayant deux subplots(pour précision et loss) figuresize=(15, 5) fig=figure et axs=ss-fig
    
    # plot l'accuracy dans le premier subplot 
    axs[0].plot( 
        range(1, len(model_history.history["accuracy"]) + 1), # axe x
        model_history.history["accuracy"], # axe y
    )
    # plot la val_accuracy dans le premier subplot 
    axs[0].plot( 
        range(1, len(model_history.history["val_accuracy"]) + 1), # axe x
        model_history.history["val_accuracy"], # axe y
    )

    axs[0].set_title("Model Accuracy") # Titre 
    axs[0].set_ylabel("Accuracy") # y label
    axs[0].set_xlabel("Epoch") # x label
    axs[0].set_xticks( # Affiche les valeurs sur chaque axe
        np.arange(1, len(model_history.history["accuracy"]) + 1),
        len(model_history.history["accuracy"]) / 10,
    )
    axs[0].legend(["train", "val"], loc="best") # La légende

    # plot le loss dans le deuxième subplot 
    axs[1].plot(
        range(1, len(model_history.history["loss"]) + 1), # axe x
        model_history.history["loss"] # axe y
    )
    axs[1].plot(
        range(1, len(model_history.history["val_loss"]) + 1), # axe x
        model_history.history["val_loss"], # axe y
    )
    axs[1].set_title("Model Loss") # Titre 
    axs[1].set_ylabel("Loss") # y label
    axs[1].set_xlabel("Epoch") # x label
    axs[1].set_xticks( # Affiche les valeurs sur chaque axe
        np.arange(1, len(model_history.history["loss"]) + 1),
        len(model_history.history["loss"]) / 10,
    )
    axs[1].legend(["train", "val"], loc="best") # La légende

    # saves and then shows results
    fig.savefig("plot.png")
    plt.show()
