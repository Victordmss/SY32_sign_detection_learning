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
from utils.utils import *
from machine_learning.utils import *
from machine_learning.config import *

# ------------- LOAD DATAS -----------------
print("Loading datas...")
datas_train = import_datas_into_dict(TRAINING_IMAGE_FILE_PATH, TRAINING_LABEL_FILE_PATH)
datas_val = import_datas_into_dict(VAL_IMAGE_FILE_PATH, VAL_LABEL_FILE_PATH)

# Dict format to store train & val datasets for each labels & classifiers
datasets = {
    "train" : {
        "danger" : {
            "X" : [],
            "Y": []
        }, 
        "interdiction": {
            "X" : [],
            "Y": []
        }, 
        "obligation": {
            "X" : [],
            "Y": []
        }, 
        "stop": {
            "X" : [],
            "Y": []
        }, 
        "ceder": {
            "X" : [],
            "Y": []
        }, 
        "frouge": {
            "X" : [],
            "Y": []
        }, 
        "forange": {
            "X" : [],
            "Y": []
        }, 
        "fvert": {
            "X" : [],
            "Y": []
        }, 
    },
    "val": {
        "danger" : {
            "X" : [],
            "Y": []
        }, 
        "interdiction": {
            "X" : [],
            "Y": []
        }, 
        "obligation": {
            "X" : [],
            "Y": []
        }, 
        "stop": {
            "X" : [],
            "Y": []
        }, 
        "ceder": {
            "X" : [],
            "Y": []
        }, 
        "frouge": {
            "X" : [],
            "Y": []
        }, 
        "forange": {
            "X" : [],
            "Y": []
        }, 
        "fvert": {
            "X" : [],
            "Y": []
        }, 
    }   
}


# ------------- CREATE DATASETS -----------------
print("Creating all datasets...")
# Fill datasets dictionnary
for classe in CLASSES:
    if classe not in ['ff', 'empty']:
        datasets["train"][classe]["X"], datasets["train"][classe]["Y"] = create_binary_classification_dataset(datas_train, classe)
        datasets["val"][classe]["X"], datasets["val"][classe]["Y"] = create_binary_classification_dataset(datas_val, classe)

# Dict format to store all classifiers
classifiers = {
    "danger" : None, 
    "interdiction": None,
    "obligation": None, 
    "stop": None,
    "ceder": None, 
    "frouge": None, 
    "forange": None, 
    "fvert": None
}

# ------------- CREATE CLASSIFIERS -----------------
print("Creating classifiers...")
for classe in CLASSES:
    if classe not in ['ff', 'empty']:
        classifiers[classe] = svm.SVC(kernel='poly')


# ------------- TRAIN & TEST CLASSIFIERS -----------------
print("Train and testing all classifiers...")
for classe in CLASSES:
    if classe not in ['ff', 'empty']:
        X_train, y_train = datasets['train'][classe]["X"], datasets['train'][classe]["Y"]
        X_val, y_val = datasets['val'][classe]["X"], datasets['val'][classe]["Y"]
        classifiers[classe].fit(X_train, y_train)
        y_pred = classifiers[classe].predict(X_val)
        print(f"Précision pour panneaux {classe}: {np.mean(y_pred == y_val)}")
