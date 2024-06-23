from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import cv2
import random

AVERAGE_SIZE = (100, 100)  # Computed with statistics
WINDOW_SIZE_SIGN = (64, 64)
WINDOW_SIZE_LIGHT = (43, 100)
AREA_THRESHOLD = 3500 # prevent to classificate too little sign/light
 
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

#  Generate an empty box for images without label
def generate_empty_bbox(image_width, image_height):
    # Generating random coords for the bbox
    x_min = random.randint(0, image_width - AVERAGE_SIZE[0])
    y_min = random.randint(0, image_height - AVERAGE_SIZE[1])
    
    # Compute complete coords of the bbox
    x_max = x_min + AVERAGE_SIZE[0]
    y_max = y_min + AVERAGE_SIZE[1]
    
    return (x_min, y_min, x_max, y_max)