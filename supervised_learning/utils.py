import os 
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv

from utils.utils import *
from supervised_learning.config import *


# Load datas into data dictionnary
def import_datas_into_dict(image_dir, label_dir):
    datas = {}
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        name = image_file.split('.')[0]
        label_path = os.path.join(label_dir, f"{name}.csv")

        image = np.array(Image.open(image_path))
        with open(label_path, 'r') as file:
            rows = file.readlines()
            label_data = {}
            if rows == ['\n']:
                for i in range(5):
                    xmin, ymin, xmax, ymax = generate_empty_bbox(image.shape[1], image.shape[0])
                    cropped_image = np.array(Image.fromarray(image[ymin:ymax, xmin:xmax]).resize(AVERAGE_SIZE))
                    label_data[i] = {
                        "name": "empty",
                        "coord": (xmin, ymin, xmax, ymax),
                        "img": cropped_image
                    }
            else:
                for i, row in enumerate(rows):
                    row = row.strip().split(",")
                    xmin, ymin, xmax, ymax = map(int, row[0:4])
                    class_name = row[4]
                    cropped_image = np.array(Image.fromarray(image[ymin:ymax, xmin:xmax]).resize(AVERAGE_SIZE))
                    label_data[i] = {
                        "name": class_name,
                        "coord": (xmin, ymin, xmax, ymax),
                        "img": cropped_image
                    }
        datas[name] = {
            "img": image,
            "labels": label_data,
        }
    return datas

# Create a binary dataset for a specific label to convert 
# the multiclass problem into several binary problems.
def create_binary_classification_dataset(datas, key):
    X = []
    Y = []

    for name, data in datas.items():
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Get ROI (region of interest)
            
            # HOG features
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
        
            # HUE features            
            color_features = np.histogram(rgb2hsv(region)[:,:,0], bins=10, range=(0, 1), density=True)[0]
            
            # Concatenate ROI features
            roi_features = np.concatenate((hog_features, color_features))

            if label["name"] == key:
                X.append(roi_features)  # Add informations into X 
                Y.append(1)  # "key" sign, classification value is 1
            else:
                X.append(roi_features)  # Add informations into X
                Y.append(0)  # Non-"key" (other classes), classification value is 0

    return np.array(X), np.array(Y)

# Function to extract all regions of interest from an image
def extract_rois_from_image(image, classifiers_dict):
    rois = []
    # Loop over the image pyramid
    for resized, scale  in pyramid(image):
        # Loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, step_size=STEP_SIZE, window_size=AVERAGE_SIZE):
            # If the window does not meet our desired window size, ignore it
            if window.shape[0] != AVERAGE_SIZE[1] or window.shape[1] != AVERAGE_SIZE[0]:
                continue
            
              # HOG features
            hog_features = np.array(hog(rgb2gray(window), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
        
            # HUE features            
            color_features = np.histogram(rgb2hsv(window)[:,:,0], bins=10, range=(0, 1), density=True)[0]
            
            # Concatenate ROI features
            roi_features = np.concatenate((hog_features, color_features)).reshape(1, -1)
            
            probas = {
                "danger" : None, 
                "interdiction": None,
                "obligation": None, 
                "stop": None,
                "ceder": None, 
                "frouge": None, 
                "forange": None, 
                "fvert": None
            }

            for classe, classifier in classifiers_dict.items():
                proba = classifier.predict_proba(roi_features)[0][1]
                if proba > 0.7:
                    probas[classe] = proba
                else:
                    probas[classe] = 0
            
            max_proba = 0
            max_classe = "empty"
            for classe, proba in probas.items():
                if proba > max_proba:
                    max_proba = proba
                    max_classe = classe
            
            
            if max_classe not in ["empty", "frouge", "fvert", "forange"]:
                x0 = int(x * scale)
                y0 = int(y * scale)
                x1 = int((x + AVERAGE_SIZE[0]) * scale)
                y1 = int((y + AVERAGE_SIZE[1]) * scale)
                rois.append([x0, y0, x1, y1, max_classe, max_proba])               
    return rois

# Function to compute a slidding window process
def sliding_window(image, step_size=STEP_SIZE, window_size=AVERAGE_SIZE):
    # Slide a window across the image
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Function to change scale of the image to compute dynamic slidding window process
def pyramid(image, scale=1.5, min_size=(30, 30)):
    # Yield the original image
    yield image, 1
    
    current_scale = 1

    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        
        # Update the current scale
        current_scale *= scale
        
        # If the resized image does not meet the supplied minimum size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
            
        # Yield the next image in the pyramid
        yield image, current_scale

# Function to compute Non Maximum Suppression process (NMS)
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

# Function that allows to display multiples rois on an image
def display_rois(image, rois):
    # Convert the image to RGB (from BGR, which is the format used by cv2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw rectangles around the ROIs
    for roi in rois:
        x0 = int(roi[0])
        y0 = int(roi[1])
        x1 = int(roi[2])
        y1 = int(roi[3])
        classe = str(roi[4])
        proba = float(roi[5])
        cv2.rectangle(image_rgb, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{classe}: {proba:.2f}"
        cv2.putText(image_rgb, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the image with the ROIs
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    