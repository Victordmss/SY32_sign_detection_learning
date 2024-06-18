from utils.utils import *
from supervised_learning.utils import *
from supervised_learning.config import *
from joblib import load

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
print("Loading classifiers...")
for classe in classifiers.keys():
    if classe not in ['ff', 'empty']:
        classifiers[classe] = load(f"{CLASSIFIERS_FOLDER_PATH}/SVM_{classe}.joblib")


# ------------- LOAD DATAS -----------------
X = {}
print("Loading validation datas...")
for filename in os.listdir(VAL_IMAGE_FOLDER_PATH):
    filepath = os.path.join(VAL_IMAGE_FOLDER_PATH, filename)
    image = Image.open(filepath)
    X[filepath] = image


# ------------- DETECTION -----------------
# Parse image with the slidding window and classify to detect labels

# Check if predicted_label folder exists
if not os.path.exists(PREDICTION_LABEL_FOLDER_PATH):
    # Create folder if doesn't exist
    os.makedirs(PREDICTION_LABEL_FOLDER_PATH)
    print(f"Folder '{PREDICTION_LABEL_FOLDER_PATH}' created with success.")

for filepath, image in X.items():
    image = np.array(image)
    name = filepath.split('/')[-1].split(".")[0]

    # Start detection process
    print(f"[DETECTION] Processing image {name}")

    # Extract rois from images with dynamic slidding window process
    rois = extract_rois_from_image(image, classifiers)

    # Filter rois with Non Maximum Suppression process
    rois = non_max_suppression(rois, iou_threshold=0.1)     
    #display_rois(image, rois)  -- UNCOMMENT TO DISPLAY

    # Write preticted labels into prediction files
    prediction_file_path = os.path.join(PREDICTION_LABEL_FOLDER_PATH, f"{name}.csv")
    with open(prediction_file_path, "w") as f:
        for roi in rois:
            x0, y0, x1, y1, classe, score = roi
            row = f"{x0},{y0},{x1},{y1},{classe},{score}\n"
            f.write(row)

    