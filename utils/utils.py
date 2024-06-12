from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import cv2
import random
import os
from PIL import Image
import csv

AVERAGE_SIZE = (32, 32)  # Thanks to the stats, we know that size of bbox will be (127, 145) -> Average size of labels 

# Dictionary for mapping class names to integers
CLASSE_TO_INT = {
    "danger": 0,
    "interdiction": 1,
    "obligation": 2,
    "stop": 3,
    "ceder": 4,
    "frouge": 5,
    "forange": 6,
    "fvert": 7,
    "ff": 8,
    "empty": 9
}

# Dictionary for mapping integers to class names
INT_TO_CLASSE = {
    0: "danger",
    1: "interdiction",
    2: "obligation",
    3: "stop",
    4: "ceder",
    5: "frouge",
    6: "forange",
    7: "fvert",
    8: "ff",
    9: "empty"
}

# Data labels key
CLASSES = ["danger", "interdiction", "obligation", "stop", "ceder", "frouge", "forange", "fvert", "ff", "empty"]

# Number of classes
NB_CLASSES = len(CLASSES)


def load_dataset(image_dir, label_dir):
    X = []  
    Y = []  
    images = []
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        file_name = int(label_file.split('.')[0]) 

        try:
            image_path = os.path.join(image_dir, str(file_name).zfill(4) + ".jpg")
            image = Image.open(image_path).convert("RGB")

            with open(label_path, "r") as file:
                reader = csv.reader(file)
                bboxes = list(reader)

            if bboxes != [[]]: 
                for box in bboxes:
                    box[4] = CLASSE_TO_INT[box[4]]
                    box[:] = map(int, box)
                    
                    # Extract ROI from image (Region of interest)
                    roi = image.crop((box[0], box[1], box[2], box[3]))  
                    # Resize with AVERAGE_SIZE
                    roi_resized = roi.resize(AVERAGE_SIZE)
                    
                    # Add to X & Y (and images) datas
                    X.append(np.array(roi_resized))
                    Y.append(box[4])
                    images.append(image)
            else:
                for _ in range(2):
                    box = list(generate_empty_bbox(image_width=image.size[1], image_height=image.size[0]))
                    
                    # Extract ROI from image (Region of interest)
                    roi = image.crop((box[0], box[1], box[2], box[3]))  
                    # Resize with AVERAGE_SIZE
                    roi_resized = roi.resize(AVERAGE_SIZE)
                    
                    # Add to X & Y (and images) datas
                    X.append(np.array(roi_resized))
                    Y.append(CLASSE_TO_INT["empty"])
                    images.append(image)

        except FileNotFoundError:
            print(f"Image file not found for {file_name}")
        except Exception as e:
            print(f"Error when processing index {file_name}: {e}")

    return np.array(X), np.array(Y)


# Function to calculate Intersection over Union (IoU) 
def iou(box1, box2):
    """
    Calcule l'Intersection over Union (IoU) entre deux boîtes englobantes.

    Parameters:
    box1 (tuple): Une boîte englobante sous la forme (x1, y1, x2, y2) où (x1, y1) est le coin supérieur gauche et (x2, y2) est le coin inférieur droit.
    box2 (tuple): Une deuxième boîte englobante sous la même forme (x1, y1, x2, y2).

    Returns:
    float: La valeur IoU entre les deux boîtes englobantes.
    """
    
    # Coordonnées des coins des boîtes ([Axe][corner_idx]_[boxe_idx])
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # Calcul des coordonnées de l'intersection
    x1_inter = max(x0_1, x0_2)
    y1_inter = max(y0_1, y0_2)
    x2_inter = min(x1_1, x1_2)
    y2_inter = min(y1_1, y1_2)

    # Calcul de l'aire de l'intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calcul de l'aire des deux boîtes
    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
    box2_area = (x1_2 - x0_2) * (y1_2 - y0_2)

    # Calcul de l'aire de l'union
    union_area = box1_area + box2_area - inter_area

    # Calcul de l'IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

# Function to calculate Non Maximum Suppression (NMS) 
def nms(bboxes, iou_threshold, score_threshold):
    """
    Applique la Non-Maximum Suppression (NMS) pour supprimer les boîtes englobantes redondantes.

    Parameters:
    bboxes (list of tuples): Une liste de tuples sous la forme (x1, y1, x2, y2, score) où (x1, y1) est le coin supérieur gauche, (x2, y2) est le coin inférieur droit et score est la confiance de la détection.
    iou_threshold (float): Le seuil d'IoU pour supprimer les boîtes redondantes.
    score_threshold (float): Le seuil de confiance pour garder les boîtes.

    Returns:
    list of tuples: Les boîtes filtrées après l'application de la NMS.
    """
    
    # Filtrer les boîtes avec un score inférieur au seuil de confiance
    bboxes = [box for box in bboxes if box[4] >= score_threshold]
    
    if len(bboxes) == 0:
        return []

    # Trier les boîtes par score de confiance décroissant
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    
    # Liste des boîtes conservées
    selected_bboxes = []

    while bboxes:
        # Prendre la boîte avec le score le plus élevé
        current_box = bboxes.pop(0)
        selected_bboxes.append(current_box)
        
        # Filtrer les boîtes restantes par IoU
        bboxes = [
            box for box in bboxes
            if iou(current_box, box) < iou_threshold
        ]
    
    return selected_bboxes

# Function to plot images with bounding boxes and class labels 
def plot_bbox_image(image, boxes):
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    
    # Getting different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, NB_CLASSES)] 

    # Getting the height and width of the image 
    h, w, _ = image.shape 

    # Create figure and axes 
    fig, ax = plt.subplots(1) 

    # Add image to plot 
    ax.imshow(image) 

    # Plotting the bounding boxes and labels over the image 
    for box in boxes:
        # Get the class from the box 
        try:
            class_pred = box[4]
        except:
            class_pred=1  # No classe (maybe because of selective search) set at 1 randomly

        x = box[0] 
        y = box[1]
        width = box[2] - x
        height = box[3] - y

        # Create a Rectangle patch with the bounding box 
        rect = patches.Rectangle( 
            (x, y), width, height, 
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
        
        # Add the patch to the Axes 
        ax.add_patch(rect) 
        
        # Add class name to the patch 
        plt.text( 
            x, 
            y, 
            s=INT_TO_CLASSE[int(class_pred)], 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        )

    # Display the plot 
    plt.show()


def selective_search(image, visualize=False, visulize_count=100):
    # Convert image to BGR format for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Initialiser la recherche sélective
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    # Utiliser la recherche sélective en mode rapide (ou en mode qualité)
    ss.switchToSelectiveSearchFast()  # Pour la recherche rapide
    # ss.switchToSelectiveSearchQuality()  # Pour une recherche plus précise

    # Obtenir les régions candidates
    roi = ss.process()

    if visualize:
        # Dessiner les régions candidates sur l'image
        for (x, y, w, h) in roi[:visulize_count]:  # Limiter à 100 régions pour la visualisation
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Afficher l'image avec les régions candidates
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    return roi


#  Generate an empty box for images without label
def generate_empty_bbox(image_width, image_height):
    # Generating random coords for the bbox
    x_min = random.randint(0, image_width - AVERAGE_SIZE[0])
    y_min = random.randint(0, image_height - AVERAGE_SIZE[1])
    
    # Compute complete coords of the bbox
    x_max = x_min + AVERAGE_SIZE[0]
    y_max = y_min + AVERAGE_SIZE[1]
    
    return (x_min, y_min, x_max, y_max)