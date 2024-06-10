from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

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
    
    # Coordonnées des coins des boîtes
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calcul des coordonnées de l'intersection
    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)

    # Calcul de l'aire de l'intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calcul de l'aire des deux boîtes
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

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

	# Reading the image with OpenCV 
	img = np.array(image) 
	# Getting the height and width of the image 
	h, w, _ = img.shape 

	# Create figure and axes 
	fig, ax = plt.subplots(1) 

	# Add image to plot 
	ax.imshow(img) 

	# Plotting the bounding boxes and labels over the image 
	for box in boxes:
		# Get the class from the box 
		class_pred = box[4] 
		
		x = box[0] * w 
		y = box[1] * h 
		width = box[2] * w - x
		height = box[3] * h - y

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


