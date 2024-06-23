from deep_learning.signClassifier import SignClassifierV2 as SignClassifier
from deep_learning.config import *
from utils.utils import *
from skimage.io import imread
from deep_learning.utils import *

# Create & train
clf = SignClassifier()
#clf.training_dataset.visualize_repartition()
clf.fit()

# Evaluate classification model
clf.evaluate_classification()

"""
# Test de la detection

image_test = np.array(imread("./data/train/images/0069.jpg"))

rois = selective_search(image_test)[:2000]

rois_images = []

for roi in rois:
    rois_images.append(image_test[roi[1]:roi[3], roi[0]:roi[2]])

predictions = clf.predict(rois_images)

filtered_rois = []

for i, prediction in enumerate(predictions):
    x0, y0, x1, y1 = rois[i]
    score = max(prediction)
    classe = INT_TO_CLASSE[np.argmax(prediction)]
    print(score, classe)
    if score > 0.8:
        filtered_rois.append([x0, y0, x1, y1, score, classe])

display_rois(image_test, filtered_rois)
"""

