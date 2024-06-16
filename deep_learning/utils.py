import numpy as np
from utils.utils import plot_bbox_image, INT_TO_CLASSE
import matplotlib.pyplot as plt
from utils.utils import selective_search

def compute_dataset_repartition(loader):
    # Dictionnaire pour compter le nombre d'occurrences de chaque classe
    class_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}

    # Parcours du dataset
    for images, boxes in loader:
        bboxes = []
        for bbox in boxes:
            bboxes.append([value.item() for value in bbox])
        
        for bbox in bboxes:
            classe = bbox[4]
            # Incrémentation du compteur pour cette classe
            class_counts[classe] += 1

    # Affichage des comptes de classe
    for classe, count in class_counts.items():
        print(f"Classe {INT_TO_CLASSE[classe]}: {count} occurrences")

    # Création de l'histogramme
    classes = [INT_TO_CLASSE[classe] for classe in class_counts.keys()]
    occurrences = list(class_counts.values())

    plt.bar(classes, occurrences)
    plt.xlabel('Classes')
    plt.ylabel('Occurrences')
    plt.title('Répartition des classes dans le dataset')
    plt.xticks(rotation=45)
    plt.show()