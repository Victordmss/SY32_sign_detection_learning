
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:15:51 2024

@author: Proprietaire
"""

import os
import numpy as np
from PIL import Image
from skimage import io, draw, color
from skimage.feature import hog
from skimage.color import rgb2hsv, rgb2gray
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.transform import pyramid_gaussian
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

AVERAGE_SIZE_IMAGE = (127, 145)

# Utilitaire pour générer une bbox vide pour les images sans label
def generate_empty_bbox(image_width, image_height):
    x_min = np.random.randint(0, image_width - AVERAGE_SIZE_IMAGE[0])
    y_min = np.random.randint(0, image_height - AVERAGE_SIZE_IMAGE[1])
    x_max = x_min + AVERAGE_SIZE_IMAGE[0]
    y_max = y_min + AVERAGE_SIZE_IMAGE[1]
    return (x_min, y_min, x_max, y_max)

# Chargement des données
def load_data(image_dir, label_dir):
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
                xmin, ymin, xmax, ymax = generate_empty_bbox(image.shape[1], image.shape[0])
                cropped_image = np.array(Image.fromarray(image[ymin:ymax, xmin:xmax]).resize(AVERAGE_SIZE_IMAGE))
                label_data[0] = {
                    "name": "empty",
                    "coord": (xmin, ymin, xmax, ymax),
                    "img": cropped_image
                }
            else:
                for i, row in enumerate(rows):
                    row = row.strip().split(",")
                    xmin, ymin, xmax, ymax = map(int, row[0:4])
                    class_name = row[4]
                    cropped_image = np.array(Image.fromarray(image[ymin:ymax, xmin:xmax]).resize(AVERAGE_SIZE_IMAGE))
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

# Dict pour convertir les noms de classe en entiers
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


######################################################################################
######################################################################################
# Création des caractéristiques HOG pour la détection

def regions_to_vectors_stop(datas):
    X = []
    Y = []

    for name, data in datas.items():
        image = data["img"]
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Récupérer la région labellisée
            # Appliquer un prétraitement si nécessaire (redimensionnement, mise à l'échelle, etc.)
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
            class_label = name_to_int[label["name"]]
            if class_label == name_to_int["stop"]:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(1)  # Panneau stop
            else:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(0)  # Non-stop (autres classes)

    return np.array(X), np.array(Y)

def regions_to_vectors_danger(datas):
    X = []
    Y = []

    for name, data in datas.items():
        image = data["img"]
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Récupérer la région labellisée
            # Appliquer un prétraitement si nécessaire (redimensionnement, mise à l'échelle, etc.)
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
            class_label = name_to_int[label["name"]]
            if class_label == name_to_int["danger"]:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(1)  # Panneau danger
            else:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(0)  # Non-danger (autres classes)

    return np.array(X), np.array(Y)

def regions_to_vectors_interdiction(datas):
    X = []
    Y = []

    for name, data in datas.items():
        image = data["img"]
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Récupérer la région labellisée
            # Appliquer un prétraitement si nécessaire (redimensionnement, mise à l'échelle, etc.)
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
            class_label = name_to_int[label["name"]]
            if class_label == name_to_int["interdiction"]:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(1)  # Panneau interdiction
            else:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(0)  # Non-interdiction (autres classes)

    return np.array(X), np.array(Y)

def regions_to_vectors_obligation(datas):
    X = []
    Y = []

    for name, data in datas.items():
        image = data["img"]
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Récupérer la région labellisée
            # Appliquer un prétraitement si nécessaire (redimensionnement, mise à l'échelle, etc.)
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
            class_label = name_to_int[label["name"]]
            if class_label == name_to_int["obligation"]:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(1)  # Panneau obligation
            else:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(0)  # Non-obligation (autres classes)

    return np.array(X), np.array(Y)

def regions_to_vectors_ceder(datas):
    X = []
    Y = []

    for name, data in datas.items():
        image = data["img"]
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Récupérer la région labellisée
            # Appliquer un prétraitement si nécessaire (redimensionnement, mise à l'échelle, etc.)
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
            class_label = name_to_int[label["name"]]
            if class_label == name_to_int["ceder"]:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(1)  # Panneau ceder
            else:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(0)  # Non-ceder (autres classes)

    return np.array(X), np.array(Y)

def regions_to_vectors_feu(datas):
    X = []
    Y = []

    for name, data in datas.items():
        image = data["img"]
        labels = data["labels"]

        for label in labels.values():
            region = label["img"]  # Récupérer la région labellisée
            # Appliquer un prétraitement si nécessaire (redimensionnement, mise à l'échelle, etc.)
            hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
            class_label = name_to_int[label["name"]]
            if class_label == name_to_int["frouge"] or class_label == name_to_int["forange"] or class_label == name_to_int["fvert"]:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(1)  # Feu
            else:
                X.append(hog_features)  # Ajouter les caractéristiques HOG
                Y.append(0)  # Non-feu (autres classes)

    return np.array(X), np.array(Y)


######################################################################################
######################################################################################
# Entraînement du modèle SVM avec les caractéristiques HOG
datas_train = load_data("dataset-main-train/train/images", "dataset-main-train/train/labels")
datas_val = load_data("dataset-main-val/val/images", "dataset-main-val/val/labels")

# Dossier contenant les images d'évaluation
test_image_folder = 'dataset-main-val/val/images'
output_folder_stop = 'result_detection_v2_stop'
output_folder_danger = 'result_detection_v2_danger'
output_folder_interdiction = 'result_detection_v2_interdiction'
output_folder_obligation = 'result_detection_v2_obligation'
output_folder_ceder = 'result_detection_v2_ceder'
output_folder_feu = 'result_detection_v2_feu'

X_train_stop, Y_train_stop = regions_to_vectors_stop(datas_train)
X_val_stop, Y_val_stop = regions_to_vectors_stop(datas_val)

X_train_danger, Y_train_danger = regions_to_vectors_danger(datas_train)
X_val_danger, Y_val_danger = regions_to_vectors_danger(datas_val)

X_train_interdiction, Y_train_interdiction = regions_to_vectors_interdiction(datas_train)
X_val_interdiction, Y_val_interdiction = regions_to_vectors_interdiction(datas_val)

X_train_obligation, Y_train_obligation = regions_to_vectors_obligation(datas_train)
X_val_obligation, Y_val_obligation = regions_to_vectors_obligation(datas_val)

X_train_ceder, Y_train_ceder = regions_to_vectors_ceder(datas_train)
X_val_ceder, Y_val_ceder = regions_to_vectors_ceder(datas_val)

X_train_feu, Y_train_feu = regions_to_vectors_feu(datas_train)
X_val_feu, Y_val_feu = regions_to_vectors_feu(datas_val)
######################################################################################
######################################################################################
# Créer et entraîner le classifieur SVM pour les panneaux stop
clf_stop = svm.SVC(kernel='poly')
clf_stop.fit(X_train_stop, Y_train_stop)

# Prédiction sur le jeu de validation
y_pred = clf_stop.predict(X_val_stop)
print(f"Taux d'erreur SVM pour panneaux stop: {np.mean(y_pred != Y_val_stop)}")

clf_danger = svm.SVC(kernel='poly')
clf_danger.fit(X_train_danger, Y_train_danger)

# Prédiction sur le jeu de validation
y_pred = clf_danger.predict(X_val_danger)
print(f"Taux d'erreur SVM pour panneaux danger: {np.mean(y_pred != Y_val_danger)}")

clf_interdiction = svm.SVC(kernel='poly')
clf_interdiction.fit(X_train_interdiction, Y_train_interdiction)

# Prédiction sur le jeu de validation
y_pred = clf_interdiction.predict(X_val_interdiction)
print(f"Taux d'erreur SVM pour panneaux interdiction: {np.mean(y_pred != Y_val_interdiction)}")

clf_obligation = svm.SVC(kernel='poly')
clf_obligation.fit(X_train_obligation, Y_train_obligation)

# Prédiction sur le jeu de validation
y_pred = clf_obligation.predict(X_val_obligation)
print(f"Taux d'erreur SVM pour panneaux obligation: {np.mean(y_pred != Y_val_obligation)}")

clf_ceder = svm.SVC(kernel='poly')
clf_ceder.fit(X_train_ceder, Y_train_ceder)

# Prédiction sur le jeu de validation
y_pred = clf_ceder.predict(X_val_ceder)
print(f"Taux d'erreur SVM pour panneaux ceder: {np.mean(y_pred != Y_val_ceder)}")

clf_feu = svm.SVC(kernel='poly')
clf_feu.fit(X_train_feu, Y_train_feu)

# Prédiction sur le jeu de validation
y_pred = clf_feu.predict(X_val_feu)
print(f"Taux d'erreur SVM pour panneaux feu: {np.mean(y_pred != Y_val_feu)}")
####################################################################################
####################################################################################
# Fonction pour faire glisser une fenêtre sur l'image
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Fonction pour la suppression des doublons (NMS)
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

# Préparation du dossier de sortie

if not os.path.exists(output_folder_stop):
    os.makedirs(output_folder_stop)
    
if not os.path.exists(output_folder_danger):
    os.makedirs(output_folder_danger)
    
if not os.path.exists(output_folder_interdiction):
    os.makedirs(output_folder_interdiction)
    
if not os.path.exists(output_folder_obligation):
    os.makedirs(output_folder_obligation)
    
if not os.path.exists(output_folder_ceder):
    os.makedirs(output_folder_ceder)
    
if not os.path.exists(output_folder_feu):
    os.makedirs(output_folder_feu)


######################################################################################
######################################################################################
#Fonction prediction?
"""
def prediction(hog_features):
    if clf_stop.predict(hog_features)==1:
        return "stop"
    if clf_danger.predict(hog_features)==1:
        return "danger"
    if clf_interdiction.predict(hog_features)==1:
        return "interdiction"
    if clf_obligation.predict(hog_features)==1:
        return "obligation"
"""
######################################################################################
######################################################################################
######################################################################################
#Detection et classification
window_sizes = [(64, 64), (128, 128), (256, 256),(512,512)]  # Différentes tailles de fenêtres
step_size = 32
stop=0
danger=0
interdiction=0
obligation=0
ceder=0

feu=0



    
    
for filename in os.listdir(test_image_folder):
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    detections = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
                
            # Extraire les caractéristiques HOG de la fenêtre
            window_resized = np.array(Image.fromarray(window).resize(AVERAGE_SIZE_IMAGE))
            hog_features = np.array(hog(rgb2gray(window_resized), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten().reshape(1, -1)

            #pred = prediction(hog_features)
            pred=clf_stop.predict(hog_features)
            
            if pred==1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                
            """
            if pred == "stop":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                stop=stop+1
            elif pred == "danger":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                danger=danger+1
            elif pred == "interdiction":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                interdiction=interdiction+1
            elif pred == "obligation":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                obligation=obligation+1
            """

    # Suppression des doublons (NMS)
    nms_boxes = non_max_suppression(detections, overlap_thresh=0.3)

    for (x1, y1, x2, y2) in nms_boxes:
        rr, cc = draw.rectangle_perimeter(start=(y1, x1), extent=(y2 - y1, x2 - x1), shape=test_image.shape)
        test_image[rr, cc] = [255, 0, 0] 
        stop=stop+1
    output_path = os.path.join(output_folder_stop, filename)
    io.imsave(output_path, test_image)
    print(f"Processed and saved: {filename}")
    
    
for filename in os.listdir(test_image_folder):
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    detections = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
                
            # Extraire les caractéristiques HOG de la fenêtre
            window_resized = np.array(Image.fromarray(window).resize(AVERAGE_SIZE_IMAGE))
            hog_features = np.array(hog(rgb2gray(window_resized), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten().reshape(1, -1)

            #pred = prediction(hog_features)
            pred=clf_danger.predict(hog_features)
            
            if pred==1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                
            """
            if pred == "stop":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                stop=stop+1
            elif pred == "danger":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                danger=danger+1
            elif pred == "interdiction":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                interdiction=interdiction+1
            elif pred == "obligation":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                obligation=obligation+1
            """

    # Suppression des doublons (NMS)
    nms_boxes = non_max_suppression(detections, overlap_thresh=0.3)

    for (x1, y1, x2, y2) in nms_boxes:
        rr, cc = draw.rectangle_perimeter(start=(y1, x1), extent=(y2 - y1, x2 - x1), shape=test_image.shape)
        test_image[rr, cc] = [255, 0, 0] 
        danger=danger+1
    output_path = os.path.join(output_folder_danger, filename)
    io.imsave(output_path, test_image)
    print(f"Processed and saved: {filename}")
    
for filename in os.listdir(test_image_folder):
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    detections = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
                
            # Extraire les caractéristiques HOG de la fenêtre
            window_resized = np.array(Image.fromarray(window).resize(AVERAGE_SIZE_IMAGE))
            hog_features = np.array(hog(rgb2gray(window_resized), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten().reshape(1, -1)

            #pred = prediction(hog_features)
            pred=clf_interdiction.predict(hog_features)
            
            if pred==1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                
            """
            if pred == "stop":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                stop=stop+1
            elif pred == "danger":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                danger=danger+1
            elif pred == "interdiction":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                interdiction=interdiction+1
            elif pred == "obligation":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                obligation=obligation+1
            """

    # Suppression des doublons (NMS)
    nms_boxes = non_max_suppression(detections, overlap_thresh=0.3)

    for (x1, y1, x2, y2) in nms_boxes:
        rr, cc = draw.rectangle_perimeter(start=(y1, x1), extent=(y2 - y1, x2 - x1), shape=test_image.shape)
        test_image[rr, cc] = [255, 0, 0] 
        interdiction=interdiction+1
    output_path = os.path.join(output_folder_interdiction, filename)
    io.imsave(output_path, test_image)
    print(f"Processed and saved: {filename}")

for filename in os.listdir(test_image_folder):
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    detections = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
                
            # Extraire les caractéristiques HOG de la fenêtre
            window_resized = np.array(Image.fromarray(window).resize(AVERAGE_SIZE_IMAGE))
            hog_features = np.array(hog(rgb2gray(window_resized), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten().reshape(1, -1)

            #pred = prediction(hog_features)
            pred=clf_obligation.predict(hog_features)
            
            if pred==1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                
            """
            if pred == "stop":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                stop=stop+1
            elif pred == "danger":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                danger=danger+1
            elif pred == "interdiction":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                interdiction=interdiction+1
            elif pred == "obligation":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                obligation=obligation+1
            """

    # Suppression des doublons (NMS)
    nms_boxes = non_max_suppression(detections, overlap_thresh=0.3)

    for (x1, y1, x2, y2) in nms_boxes:
        rr, cc = draw.rectangle_perimeter(start=(y1, x1), extent=(y2 - y1, x2 - x1), shape=test_image.shape)
        test_image[rr, cc] = [255, 0, 0] 
        obligation=obligation+1
    output_path = os.path.join(output_folder_obligation, filename)
    io.imsave(output_path, test_image)
    print(f"Processed and saved: {filename}")
    
for filename in os.listdir(test_image_folder):
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    detections = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
                
            # Extraire les caractéristiques HOG de la fenêtre
            window_resized = np.array(Image.fromarray(window).resize(AVERAGE_SIZE_IMAGE))
            hog_features = np.array(hog(rgb2gray(window_resized), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten().reshape(1, -1)

            #pred = prediction(hog_features)
            pred=clf_ceder.predict(hog_features)
            
            if pred==1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                
            """
            if pred == "stop":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                stop=stop+1
            elif pred == "danger":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                danger=danger+1
            elif pred == "interdiction":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                interdiction=interdiction+1
            elif pred == "obligation":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                obligation=obligation+1
            """

    # Suppression des doublons (NMS)
    nms_boxes = non_max_suppression(detections, overlap_thresh=0.3)

    for (x1, y1, x2, y2) in nms_boxes:
        rr, cc = draw.rectangle_perimeter(start=(y1, x1), extent=(y2 - y1, x2 - x1), shape=test_image.shape)
        test_image[rr, cc] = [255, 0, 0] 
        ceder=ceder+1
    output_path = os.path.join(output_folder_ceder, filename)
    io.imsave(output_path, test_image)
    print(f"Processed and saved: {filename}")
    
    
for filename in os.listdir(test_image_folder):
    test_image_path = os.path.join(test_image_folder, filename)
    test_image = io.imread(test_image_path)

    detections = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(test_image, step_size=step_size, window_size=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
                
            # Extraire les caractéristiques HOG de la fenêtre
            window_resized = np.array(Image.fromarray(window).resize(AVERAGE_SIZE_IMAGE))
            hog_features = np.array(hog(rgb2gray(window_resized), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten().reshape(1, -1)

            #pred = prediction(hog_features)
            pred=clf_feu.predict(hog_features)
            
            if pred==1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                
            """
            if pred == "stop":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                stop=stop+1
            elif pred == "danger":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                danger=danger+1
            elif pred == "interdiction":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                interdiction=interdiction+1
            elif pred == "obligation":  
                detections.append((x, y, x + window_size[0], y + window_size[1]))
                obligation=obligation+1
            """

    # Suppression des doublons (NMS)
    nms_boxes = non_max_suppression(detections, overlap_thresh=0.3)

    for (x1, y1, x2, y2) in nms_boxes:
        rr, cc = draw.rectangle_perimeter(start=(y1, x1), extent=(y2 - y1, x2 - x1), shape=test_image.shape)
        test_image[rr, cc] = [255, 0, 0] 
        feu=feu+1
    output_path = os.path.join(output_folder_feu, filename)
    io.imsave(output_path, test_image)
    print(f"Processed and saved: {filename}")



    
print("Panneaux stop détectés :", stop)
print("Panneaux danger détectés :", danger)
print("Panneaux interdiction détectés :", interdiction)
print("Panneaux obligation détectés :", obligation)
print("Panneaux ceder détectés :", ceder)
print("Panneaux feu détectés :", feu)



