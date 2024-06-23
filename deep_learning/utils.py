import numpy as np
import os 
import matplotlib.pyplot as plt
from PIL import Image
import csv
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

# Function that process a selective search on the current image
def selective_search(image):
    regions = []
    
    segments = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    segments.setBaseImage(image)
    segments.switchToSelectiveSearchFast() 

    rects = segments.process()

    for (x, y, w, h) in rects:
        regions.append((x, y, x + w, y + h))  
    
    return regions

def non_max_suppression(rois, iou_threshold=0.1):
    """
    Apply Non-Maximum Suppression to avoid multiple detections of the same object.
    
    Parameters:
    - rois: List of ROIs where each ROI is a list [x0, y0, x1, y1, classe, score]
    - iou_threshold: Threshold for Intersection over Union (IoU) to suppress overlapping boxes
    
    Returns:
    - List of ROIs after NMS
    """
    if len(rois) == 0:
        return []
    
    rois = np.array(rois)
    
    # Coordinates of bounding boxes
    x0 = rois[:, 0].astype(int)
    y0 = rois[:, 1].astype(int)
    x1 = rois[:, 2].astype(int)
    y1 = rois[:, 3].astype(int)
    scores = rois[:, 5].astype(float)

    # Compute the area of the bounding boxes and sort the bounding boxes by the score
    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute the intersection area
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])

        w = np.maximum(0, xx1 - xx0 + 1)
        h = np.maximum(0, yy1 - yy0 + 1)
        intersection = w * h

        # Compute the IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep only the boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return rois[keep].tolist()

    
# Function that compute Union on Intersection value betwen 2 bbox
def calculate_iou(bbox1, bbox2):
    """
    Calcule l'Intersection over Union (IoU) entre deux boîtes englobantes.

    Arguments :
    bbox1, bbox2 -- listes ou tuples de format [x0, y0, x1, y1]

    Retourne :
    iou -- Intersection over Union (IoU) en tant que flottant
    """

    # Coordonnées de l'intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Calculer l'aire de l'intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # Pas d'intersection

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Aires des boîtes englobantes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Aire de l'union
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calcul de l'IoU
    iou = intersection_area / union_area

    return iou


def display_rois(image, rois):    
    # Draw rectangles around the ROIs
    for roi in rois:
        x0 = int(roi[0])
        y0 = int(roi[1])
        x1 = int(roi[2])
        y1 = int(roi[3])
        classe = str(roi[4])
        proba = float(roi[5])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{classe}: {proba:.2f}"
        cv2.putText(image, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the image with the ROIs
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()