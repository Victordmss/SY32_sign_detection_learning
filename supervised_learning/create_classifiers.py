import numpy as np
from sklearn import svm

from utils.utils import *
from supervised_learning.utils import *
from supervised_learning.config import *
from joblib import dump

# ------------- LOAD DATAS -----------------
print("Loading datas...")
datas_train = import_datas_into_dict(TRAINING_IMAGE_FOLDER_PATH, TRAINING_LABEL_FOLDER_PATH)
datas_val = import_datas_into_dict(VAL_IMAGE_FOLDER_PATH, VAL_LABEL_FOLDER_PATH)

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
        "feux" : {
             "X": [],
             "Y": []
        }
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
        "feux" : {
             "X": [],
             "Y": []
        }
    }   
}


# ------------- CREATE DATASETS -----------------
print("Creating all datasets...")
# Fill datasets dictionnary
for classe in CLASSES:
    if classe not in ['ff', 'empty']:
        datasets["train"][classe]["X"], datasets["train"][classe]["Y"] = create_binary_classification_dataset(datas_train, classe)
        datasets["val"][classe]["X"], datasets["val"][classe]["Y"] = create_binary_classification_dataset(datas_val, classe)
        

# Create a dataset for general light classifier
datasets["train"]["feux"]["X"], datasets["train"]["feux"]["Y"] = create_binary_classification_dataset(datas_train, "feux")
datasets["val"]["feux"]["X"], datasets["val"]["feux"]["Y"] = create_binary_classification_dataset(datas_val, "feux")

# Dict format to store all classifiers
classifiers = {
    "danger" : None, 
    "interdiction": None,
    "obligation": None, 
    "stop": None,
    "ceder": None, 
    "frouge": None, 
    "forange": None, 
    "fvert": None,
    "feux": None
}

# ------------- CREATE CLASSIFIERS -----------------
print("Creating classifiers...")
for classe in classifiers.keys():
    if classe not in ['ff', 'empty']:
        classifiers[classe] = svm.SVC(kernel='poly', probability=True)


# ------------- TRAIN & TEST CLASSIFIERS -----------------
print("Train and testing all classifiers...")

for classe in classifiers.keys():
    if classe not in ['ff', 'empty']:
        X_train, y_train = datasets['train'][classe]["X"], datasets['train'][classe]["Y"]
        X_val, y_val = datasets['val'][classe]["X"], datasets['val'][classe]["Y"]
        classifiers[classe].fit(X_train, y_train)
        y_pred = classifiers[classe].predict(X_val)
        print(f"Précision pour panneaux {classe}: {np.mean(y_pred == y_val)}")

# ------------- SAVE CLASSIFIERS -----------------
print("Saving classifiers")
for classes, model in classifiers.items():
        dump(model, f'{CLASSIFIERS_FOLDER_PATH}/SVM_{classes}.joblib')