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
import torch

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
                    roi_resized = roi.resize(AVERAGE_SIZE)
                    
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
                    roi_resized = roi.resize(AVERAGE_SIZE)
                    
                    # Append the resized ROI to X and the class label for empty to Y
                    X.append(np.array(roi_resized))
                    Y.append(CLASSE_TO_INT["empty"])

        except FileNotFoundError:
            print(f"Image file not found for {file_name}")
        except Exception as e:
            print(f"Error when processing index {file_name}: {e}")

    # Convert the lists X and Y to numpy arrays and return them
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

#NMS implementation in Python and Numpy
def nms(bboxes, threshold=0.5):
    '''
    NMS: first sort the bboxes by scores , 
        keep the bbox with highest score as reference,
        iterate through all other bboxes, 
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.
        
    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.
        
    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
    '''
    #Unstacking Bounding Box Coordinates
    bboxes = np.array(bboxes).astype('float')
    x0 = bboxes[:,0]
    y0 = bboxes[:,1]
    x1 = bboxes[:,2]
    y1 = bboxes[:,3]
    scores = bboxes[:,5]
    
    #Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = scores.argsort()[::-1]
    #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x1-x0+1)*(y1-y0+1)
    
    #list to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x0[rbbox_i],x0[sorted_idx[1:]])
        overlap_ymins = np.maximum(y0[rbbox_i],y0[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x1[rbbox_i],x1[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y1[rbbox_i],y1[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
        
    
    #Return filtered bboxes
    return bboxes[filtered]

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

        if class_pred not in  [CLASSE_TO_INT["empty"]]:
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