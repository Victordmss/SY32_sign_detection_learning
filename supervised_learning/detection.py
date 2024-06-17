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
X = []
print("Loading validation datas...")
for filename in os.listdir(VAL_IMAGE_FOLDER_PATH):
    filepath = os.path.join(VAL_IMAGE_FOLDER_PATH, filename)
    image = Image.open(filepath)
    X.append(image)



def sliding_window(image, step_size, window_size):
    # Slide a window across the image
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def pyramid(image, scale=1.5, min_size=(30, 30)):
    # Yield the original image
    yield image
    
    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        
        # If the resized image does not meet the supplied minimum size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
            
        # Yield the next image in the pyramid
        yield image

step_size = 16

# ------------- DETECTION -----------------
# Parse image with the slidding window and classify to detect labels
for image in X:
    image = np.array(image)
    # Loop over the image pyramid
    for resized in pyramid(image):
        # Loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=AVERAGE_SIZE):
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

            for classe, classifier in classifiers.items():
                proba = classifier.predict_proba(roi_features)[0][1]
                if proba > 0.6:
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
                plt.imshow(window)
                plt.show()
                print(max_classe)

# TO DO : NMS + FAUX NEGATIS TRAINING

# RAPPEL :
# STOP : BON
# OBLIGATION : OK
# DANGER : NUL
# FEUX : NUL
# INTERDICTION : CONFOND AVEC STOP (MAIS PAS DEGEUX NON PLUS)
# CEDER : PAS FOU CONFOND AVEC INTERDICTION ET ARBRES PARFOIS
