import numpy as np
import os 
import matplotlib.pyplot as plt
from PIL import Image

from utils.utils import *
from utils.utils import *
from deep_learning.config import *
from utils.utils import *

# Function to load the dataset into data dict
def datas_to_XY_dataset(image_dir, label_dir):
    # Initialize empty lists to store images (X) and labels (Y)
    X = []  
    Y = []  
    
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        
        file_name = int(label_file.split('.')[0])  # Extract the file name to find corresponding image file

        try:
            image_path = os.path.join(image_dir, str(file_name).zfill(4) + ".jpg")            
            image = Image.open(image_path).convert("RGB")  # Open the image

            # Read bounding boxes from the label file
            with open(label_path, "r") as file:
                reader = csv.reader(file)
                bboxes = list(reader)

            # Check if there are any bounding boxes in the label file
            if bboxes != [[]]: 
                # Iterate over each bounding box
                for box in bboxes:
                    # Convert class label from string to integer using CLASSE_TO_INT dictionary
                    box[4] = CLASSE_TO_INT[box[4]]
                    # Convert all elements in the bounding box to integers
                    box[:] = map(int, box)
                    
                    # Extract Region of Interest (ROI) from the image based on the bounding box
                    roi = image.crop((box[0], box[1], box[2], box[3]))  
                    # Resize the ROI to a predefined average size
                    roi_resized = roi.resize(RESIZE_SIZE)
                    
                    # Append the resized ROI to X and its corresponding class label to Y
                    X.append(np.array(roi_resized))
                    Y.append(box[4])
                    
            else:
                # If no bounding boxes are present, generate empty bounding boxes
                for _ in range(5):
                    box = list(generate_empty_bbox(image_width=image.size[1], image_height=image.size[0]))
                    
                    # Extract ROI from image based on empty bounding box
                    roi = image.crop((box[0], box[1], box[2], box[3]))  
                    # Resize the ROI to a predefined average size
                    roi_resized = roi.resize(RESIZE_SIZE)
                    
                    # Append the resized ROI to X and the class label for empty to Y
                    X.append(np.array(roi_resized))
                    Y.append(CLASSE_TO_INT["empty"])

        except FileNotFoundError:
            print(f"Image file not found for {file_name}")
        except Exception as e:
            print(f"Error when processing index {file_name}: {e}")

    # Convert the lists X and Y to numpy arrays and return them
    return np.array(X), np.array(Y)

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