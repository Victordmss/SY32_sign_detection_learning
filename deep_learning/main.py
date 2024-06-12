from deep_learning.signClassifier import SignClassifier
from deep_learning.config import *
from utils.utils import *
from skimage.io import imread

# Create & train
clf = SignClassifier()
#clf.training_dataset.visualize_repartition()
clf.fit()

# Evaluate classification model
clf.evaluate_classification()


image_test = np.array(imread("./data/train/images/0069.jpg"))
clf.predict(image_test, visualize=True)



