from utils.utils import *
from supervised_learning.utils import *
from supervised_learning.config import *
from joblib import load, dump
import argparse 
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv

# ------------- IMPORT PARSER & ARGS ---------------

parser = argparse.ArgumentParser()
parser.add_argument('input_file', metavar='DIR', help='Folder of input images to analyse')

# Analyser les arguments de la ligne de commande
args = parser.parse_args()

# Load predicted label folder folder
input_file = args.input_file

print(f"[TRAIN FP] Start re-training on false positives with {input_file} results")

# ------------- LOAD CLASSIFIERS -----------------
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

# Parse dict and load all classifiers
print("[TRAIN FP] Loading classifiers...")
for classe in classifiers.keys():
    if classe not in ['ff', 'empty']:
        classifiers[classe] = load(f"{CLASSIFIERS_FOLDER_PATH}/SVM_{classe}.joblib")

# ------------- LOAD DATAS -----------------
print("[TRAIN FP] Loading input datas...")

# 1. Load real labels
real_labels = {}   # Dict with all the real label linked with the image name
# Load real labels
for filename in os.listdir(TRAINING_LABEL_FOLDER_PATH):
    name = filename.split(".")[0]
    filepath = os.path.join(TRAINING_LABEL_FOLDER_PATH, filename)

    real_labels[name] = []
    with open(filepath, "r") as label_file:
        rows = label_file.readlines()
        if rows != ['\n']:
            for row in rows:
                row = row.strip().split(",")
                xmin, ymin, xmax, ymax = map(int, row[0:4])
                class_name = row[4]
                real_labels[name].append([xmin, ymin, xmax, ymax, class_name])
                    
# 2. Load images
images = {}
for filename in os.listdir(TRAINING_IMAGE_FOLDER_PATH):
    name = filename.split(".")[0]
    filepath = os.path.join(TRAINING_IMAGE_FOLDER_PATH, filename)
    image = Image.open(filepath)
    images[name] = np.array(image)


# 3. Load predicted labels
predicted_labels = {} # Dict with all the label predicted linked with the image name
# Load predicted datas
with open(input_file, "r") as label_file:
    for row in label_file.readlines():
        row = row.split(",")
        predicted_labels[row[0]].append([int(row[1]), int(row[2]), int(row[3]), int(row[4]), row[5]])


# ------------- ANALYSIS OF THE PREDICTIONS -----------------
new_training_labels = {}


for key in real_labels.keys():
    labels = real_labels[key]
    prediction = predicted_labels[key]
    new_labels = labels.copy()    
    for bbox_predicted in prediction:
        for bbox_real in labels:
            founded = False
            if calculate_iou(bbox_real[:4], bbox_predicted[:4]) > 0.5 and bbox_predicted[4]==bbox_real[4]:
                founded = True

        if not founded:
            new_labels.append([bbox_predicted[0], bbox_predicted[1], bbox_predicted[2], bbox_predicted[3], f"FP_{bbox_predicted[4]}"])
            
    new_training_labels[key] = new_labels


# ------------- CREATION OF DATASETS -----------------
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


for name, labels in new_training_labels.items():
    for label in labels:
            region = np.array(Image.fromarray(images[name][label[1]:label[3], label[0]:label[2]]).resize(AVERAGE_SIZE_SIGN))
            try:
                # HOG features
                hog_features = np.array(hog(rgb2gray(region), pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')).flatten()
            
                # HUE features            
                color_features = np.histogram(rgb2hsv(region)[:,:,0], bins=10, range=(0, 1), density=True)[0]
            except:
                pass
            
            # Concatenate ROI features
            roi_features = np.concatenate((hog_features, color_features))

            if label[4] in ["FP_danger", "FP_interdiction", "FP_obligation", "FP_stop", "FP_ceder", "FP_frouge", "FP_forange", "FP_fvert"]:
                datasets["train"][label[4][3:]]["X"].append(roi_features)
                datasets["train"][label[4][3:]]["Y"].append(0)
            else:
                for classe in CLASSES:
                    if classe not in ["ff", "empty"]:
                        if classe == label[4]:
                            datasets["train"][classe]["X"].append(roi_features)
                            datasets["train"][classe]["Y"].append(1)
                        else:
                            datasets["train"][classe]["X"].append(roi_features)
                            datasets["train"][classe]["Y"].append(0)


# ------------- TRAIN & TEST CLASSIFIERS -----------------

datas_val = import_datas_into_dict(VAL_IMAGE_FOLDER_PATH, VAL_LABEL_FOLDER_PATH)
for classe in CLASSES:
    if classe not in ["ff", "empty"]:
        datasets["val"][classe]["X"], datasets["val"][classe]["Y"] = create_binary_classification_dataset(datas_val, classe)

print("Train and testing all classifiers...")
for classe in CLASSES:
    if classe not in ['ff', 'empty']:
        X_train, y_train = datasets["train"][classe]["X"], datasets["train"][classe]["Y"]
        X_val, y_val = datasets['val'][classe]["X"], datasets['val'][classe]["Y"]
        classifiers[classe].fit(X_train, y_train)
        y_pred = classifiers[classe].predict(X_val)
        print(f"Précision pour panneaux {classe}: {np.mean(y_pred == y_val)}")


# ------------- SAVE CLASSIFIERS -----------------
print("Saving classifiers")
for classes, model in classifiers.items():
        dump(model, f'{CLASSIFIERS_FOLDER_PATH}/SVM_{classes}.joblib')