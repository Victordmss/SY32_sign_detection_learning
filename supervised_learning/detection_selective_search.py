from utils.utils import *
from supervised_learning.utils import *
from supervised_learning.config import *
from joblib import load
import argparse 

# ------------- IMPORT PARSER & ARGS ---------------

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', metavar='DIR', help='Folder of input images to analyse')
parser.add_argument('output_file', metavar='DIR', help='Folder of output labels from the analysis')

# Analyser les arguments de la ligne de commande
args = parser.parse_args()

# Load input folder
input_folder = args.input_folder
# Check if input folder exists
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"[Error] Folder '{input_folder}' does not exist.")

# Load output file 
output_file = args.output_file

print(f"[PREDICTION] Start prediction and save results into {output_file}")

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
    "fvert": None,
    "feux":None,
}

# Parse dict and load all classifiers
print("[DETECTION] Loading classifiers...")
for classe in classifiers.keys():
    classifiers[classe] = load(f"{CLASSIFIERS_FOLDER_PATH}/SVM_{classe}.joblib")


# ------------- LOAD DATAS -----------------
X = {}
print("[DETECTION] Loading input datas...")
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    image = Image.open(filepath)
    X[filepath] = image


# ------------- DETECTION -----------------
# Parse image with the slidding window and classify to detect labels
# Open output file once before the loop
with open(output_file, "w") as f:
    # Parse image with the sliding window and classify to detect labels
    print(f"[DETECTION] Starting detection process")
    for filepath, image in X.items():
        image = np.array(image)

        if '/' in filepath:
            name = filepath.split('/')[-1].split(".")[0]
        else:
            name = filepath.split('\\')[-1].split(".")[0]

        # Start detection process
        print(f"[DETECTION] Processing image {name}")

        # 1. Extract rois from images with dynamic sliding window process
        rois = []


        rois = selective_search(image)
        for roi in rois[:200]:
            x0, y0, x1, y1 = roi
            window = np.array(Image.fromarray(image[y0:y1, x0:x1]).resize(AVERAGE_SIZE))

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
            }

            for classe, classifier in classifiers.items():
                if classe not in ["feux", "fvert", "frouge", "forange"]:
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
            
            # Classificate if there is a light
            proba = classifiers["feux"].predict_proba(roi_features)[0][1]
            if proba > 0.95 and proba > max_proba:
                max_proba = proba
                max_classe = "feux"

            if max_classe != "empty":
                rois.append([x0, y0, x1, y1, max_classe, max_proba])             

        # 2. Filter rois with Non Maximum Suppression process
        rois = non_max_suppression(rois, iou_threshold=0.1)     
        #display_rois(image, rois)

        # 3. Write predicted labels into prediction files
        for roi in rois:
            x0, y0, x1, y1, classe, score = roi
            row = f"{name}, {x0},{y0},{x1},{y1},{score},{classe}\n"
            f.write(row)

print(f"[PREDICTION] Results saved into {output_file}")
