# Configuration of machine learning process

TRAINING_IMAGE_FOLDER_PATH = "./data/train/images/"
TRAINING_LABEL_FOLDER_PATH = "./data/train/labels/"
VAL_IMAGE_FOLDER_PATH = "./data/val/images/"
VAL_LABEL_FOLDER_PATH = "./data/val/labels/"
PREDICTION_LABEL_FOLDER_PATH = "./data/train/predicted_labels/"
CLASSIFIERS_FOLDER_PATH = "./supervised_learning/classifiers/saves/"
WINDOW_SIZES = [(64, 64), (128, 128), (256, 256),(512,512)]  # Window sizes during slidding window process
STEP_SIZE = 16