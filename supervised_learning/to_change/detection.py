# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:46:03 2024

@author: Proprietaire
"""

import os
import numpy as np
import random
from PIL import Image
from skimage import color, feature, io

AVERAGE_SIZE_IMAGE = (127, 145)  # Thanks to the stats, we know that size of bbox will be (127, 145) -> Average size of labels 

def generate_empty_bbox(image_width, image_height):
    """ 
    Generate an empty box for images without label
    """
    # Generating random coords for the bbox
    x_min = random.randint(0, image_width - AVERAGE_SIZE_IMAGE[0])
    y_min = random.randint(0, image_height - AVERAGE_SIZE_IMAGE[1])
    
    # Compute complete coords of the bbox
    x_max = x_min + AVERAGE_SIZE_IMAGE[0]
    y_max = y_min + AVERAGE_SIZE_IMAGE[1]
    
    return (x_min, y_min, x_max, y_max)

def load_data(image_dir, label_dir):
    """ 
    Create a dict with all the usefull datas of the dataset
    datas = {
        "XXXX" (name of the file) : {
            "img" : image as an array,
            "labels" (data of the labels): {
                "X" index of the label (0,1,...,n) : {
                    "name" : name of the label,
                    "coord" : coord of the label like xmin, ymin, xmax, ymax,
                    "img" : crooped img of the label,
                }
            }
        }
    }
    """
    
    datas = {}

    for image_file in os.listdir(image_dir):
        # Computing name and files paths
        image_path = image_dir + '/' + image_file
        name = image_file.split('.')[0]
        label_path = label_dir + '/' + name + '.csv'
        
        # Import image as array
        image = np.array(Image.open(image_path))

        # Import labels as array 
        with open(label_path, 'r') as file:
            rows = file.readlines()

            label_data = {}
            if rows == ['\n']:  # Create a random empty label to balance model
                # Create random coords for empty label
                xmin, ymin, xmax, ymax = generate_empty_bbox(image.shape[1], image.shape[0])
    
                # Get the cropped image (as array) of the label
                cropped_image = np.array(Image.fromarray(image[ymin:ymax, xmin:xmax]).resize(AVERAGE_SIZE_IMAGE))
               
                label_data[0] = {
                        "name":"empty",
                        "coord": (xmin, ymin, xmax, ymax),
                        "img":cropped_image
                    }
            else:
                for i, row in enumerate(rows):  # One image can contain several labels
                    row = row.strip().split(",")

                    # Compute coords of the label
                    xmin, ymin, xmax, ymax = map(int, row[0:4])

                    # Get the label name
                    class_name = row[4]

                    # Get the cropped image (as array) of the label
                    cropped_image = np.array(Image.fromarray(image[ymin:ymax, xmin:xmax]).resize(AVERAGE_SIZE_IMAGE))
                    
                    # Adding to the json
                    label_data[i] = {
                        "name":class_name,
                        "coord": (xmin, ymin, xmax, ymax),
                        "img":cropped_image
                    }

        datas[name] = {
             "img" : image,
             "labels" : label_data,
        }
       
    return datas

# Dict to convert str class name to int
name_to_int = {
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
def create_xy(datas):
    # Creating arrays with all labels datas & classes
    X = []
    Y = []

    for name, data in datas.items():
        for row in data["labels"].values():
            image_as_array = np.array(row["img"]).flatten()
            X.append(image_as_array)
            Y.append(name_to_int[row["name"]])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


# Training dataset
datas_train = load_data("dataset-main-train/train/images", "dataset-main-train/train/labels")
X_train, Y_train = create_xy(datas=datas_train)

# Validation dataset
datas_val = load_data("dataset-main-val/val/images", "dataset-main-val/val/labels")
X_val, Y_val = create_xy(datas=datas_val)


from skimage.feature import hog
from skimage.color import rgb2gray

def extract_hog(datas):
    # Creating X array with all HOG information of images
    X = []

    for name, data in datas.items():
        for row in data["labels"].values():
            image_as_array = np.array(hog(rgb2gray(row["img"]))).flatten()
            X.append(image_as_array)

    return np.array(X)


# Update training dataset
X_train_HOG = extract_hog(datas=datas_train)

# Update validation dataset
X_val_HOG = extract_hog(datas=datas_val)


from skimage.color import rgb2hsv

def extract_color_features(datas):
    # Creating X array with all HOG information of images
    X = []

    for name, data in datas.items():
        for row in data["labels"].values():
            # Convertir l'image en espace colorimétrique HSV
            hsv_image = rgb2hsv(row["img"])

            # Calculer l'histogramme de couleur pour chaque canal
            hue_hist = np.histogram(hsv_image[:,:,0], bins=10, range=(0, 1), density=True)[0]
            saturation_hist = np.histogram(hsv_image[:,:,1], bins=10, range=(0, 1), density=True)[0]
            value_hist = np.histogram(hsv_image[:,:,2], bins=10, range=(0, 1), density=True)[0]

            # Concaténer les histogrammes de couleur
            color_features = np.concatenate((hue_hist, saturation_hist, value_hist))

            X.append(color_features)

    return np.array(X)


from sklearn import svm
from skimage import img_as_float, draw

# Update training dataset
X_train_COLORS = extract_color_features(datas=datas_train)

# Update validation dataset
X_val_COLORS = extract_color_features(datas=datas_val)

X_train_combined = np.concatenate((X_train_HOG, X_train_COLORS), axis=1)
X_val_combined = np.concatenate((X_val_HOG, X_val_COLORS), axis=1)

clf = svm.SVC(kernel='poly') 
clf.fit(X_train_combined, Y_train)
y_combined = clf.predict(X_val_combined)


from skimage.draw import rectangle

# Fonction pour faire glisser une fenêtre sur l'image
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Dossier contenant les images de test
test_image_folder = 'dataset-main-train/train/images'
output_folder = 'result_detection'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

window_size = AVERAGE_SIZE_IMAGE  # Taille de la fenêtre
step_size = 32  # Pas de la fenêtre

for filename in os.listdir(test_image_folder):
    # Charger l'image de test
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    # Faire glisser la fenêtre et détecter les feux rouges
    for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue
            
        # Extraire les caractéristiques HOG et de couleur de la fenêtre
        hog_features = np.array(hog(rgb2gray(window))).flatten().reshape(1, -1)
        hsv_image = rgb2hsv(window)
        hue_hist = np.histogram(hsv_image[:,:,0], bins=10, range=(0, 1), density=True)[0]
        saturation_hist = np.histogram(hsv_image[:,:,1], bins=10, range=(0, 1), density=True)[0]
        value_hist = np.histogram(hsv_image[:,:,2], bins=10, range=(0, 1), density=True)[0]
        color_features = np.concatenate((hue_hist, saturation_hist, value_hist)).reshape(1, -1)

        # Combiner les caractéristiques HOG et de couleur
        combined_features = np.concatenate((hog_features, color_features), axis=1)
            
        # Prédire avec le classificateur
        prediction = clf.predict(combined_features)
        print(prediction)
        if prediction == 0: #danger
                # Dessiner un rectangle autour de la fenêtre 
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [165, 165, 0]  # Colorier en jaune
        
        """
        if prediction == 1: #interdiction
                # Dessiner un rectangle autour de la fenêtre 
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [255, 0, 255]  # Colorier en rose
        """
                
        if prediction == 2: #obligation
                # Dessiner un rectangle autour de la fenêtre (utiliser skimage.draw pour dessiner)
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [0, 0, 255]  # Colorier en bleu
                
        if prediction == 3: #stop
                # Dessiner un rectangle autour de la fenêtre (utiliser skimage.draw pour dessiner)
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [100, 0, 0]  # Colorier en rouge sombre
                
        if prediction == 4: #ceder
                # Dessiner un rectangle autour de la fenêtre (utiliser skimage.draw pour dessiner)
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [255, 255, 255]  # Colorier en blanc
            
        if prediction == 5: #feu rouge
                # Dessiner un rectangle autour de la fenêtre (utiliser skimage.draw pour dessiner)
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [255, 0, 0]  # Colorier en rouge
                
        if prediction == 6: #feu orange
                # Dessiner un rectangle autour de la fenêtre (utiliser skimage.draw pour dessiner)
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [255, 165, 0]  # Colorier en orange
                
        if prediction == 7: #feu vert
                # Dessiner un rectangle autour de la fenêtre (utiliser skimage.draw pour dessiner)
                rr, cc = rectangle(start=(y, x), extent=(window_size[1], window_size[0]), shape=test_image.shape)
                test_image[rr, cc] = [0, 255, 0]  # Colorier en vert
                
                
        

        # Enregistrer l'image avec les feux rouges détectés
        output_path = os.path.join(output_folder, filename)
        io.imsave(output_path, test_image)
        print(f"Processed and saved: {filename}")


