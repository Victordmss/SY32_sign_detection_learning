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

# Function to compute a slidding window process
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Function to compute Non Maximum Suppression process (NMS)
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int")
