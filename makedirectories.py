import os


def makeDir():
    """Créer les dossiers dont on va stocker le dataset"""

    outer_names = ["test", "train"]
    inner_names = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "sad",
        "surprised",
        "neutral",
    ]
    os.makedirs("datack", exist_ok=True) # créer le dossier data sans génerer d'erreur su=i celui*ci existe déjà
    # De meme
    for outer_name in outer_names: 
        os.makedirs(os.path.join("datack", outer_name), exist_ok=True)
        for inner_name in inner_names:
            os.makedirs(os.path.join("datack", outer_name, inner_name), exist_ok=True)


    # Pour faire le prepocessing de ck+ : changer le code de la ligne 17 -> 22 par le code suivant
    # os.makedirs("datack", exist_ok=True) # créer le dossier data sans génerer d'erreur su=i celui*ci existe déjà
    # # De meme
    # for outer_name in outer_names: 
    #     os.makedirs(os.path.join("datack", outer_name), exist_ok=True)
    #     for inner_name in inner_names:
    #         os.makedirs(os.path.join("datack", outer_name, inner_name), exist_ok=True)
