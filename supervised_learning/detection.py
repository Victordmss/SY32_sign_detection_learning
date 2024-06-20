from utils.utils import *
from supervised_learning.utils import *
from supervised_learning.config import *
from joblib import load
import argparse 

# ------------- IMPORT PARSER & ARGS ---------------

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', metavar='DIR', help='Folder of input images to analyse')
parser.add_argument('output_folder', metavar='DIR', help='Folder of output labels from the analysis')

# Analyser les arguments de la ligne de commande
args = parser.parse_args()


# Load input folder
input_folder = args.input_folder
# Check if input folder exists
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"[Error] Folder '{input_folder}' does not exist.")

# Load output folder 
output_folder = args.output_folder
# Check if output folder exists
if not os.path.exists(output_folder):
    # Create folder if doesn't exist
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' created with success.")

print(f"[PREDICTION] Start prediction with {input_folder} folder")

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
print("[DETECTION] Loading classifiers...")
for classe in classifiers.keys():
    if classe not in ['ff', 'empty']:
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
print(f"[DETECTION] Starting detection process, output will be created in {output_folder} folder")
for filepath, image in X.items():
    image = np.array(image)

    if '/' in filepath:
        name = filepath.split('/')[-1].split(".")[0]
    else:
        name = filepath.split('\\')[-1].split(".")[0]

    # Start detection process
    print(f"[DETECTION] Processing image {name}")

    # Extract rois from images with dynamic slidding window process
    rois = extract_rois_from_image(image, classifiers)

    # Filter rois with Non Maximum Suppression process
    rois = non_max_suppression(rois, iou_threshold=0.1)     
    #display_rois(image, rois)  #-- UNCOMMENT TO DISPLAY

    # Write preticted labels into prediction files
    prediction_file_path = os.path.join(output_folder, f"{name}.csv")
    with open(prediction_file_path, "w") as f:
        for roi in rois:
            x0, y0, x1, y1, classe, score = roi
            row = f"{x0},{y0},{x1},{y1},{classe},{score}\n"
            f.write(row)

    