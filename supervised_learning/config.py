# Configuration of machine learning process

TRAINING_IMAGE_FILE_PATH = "./data/train/images/"
TRAINING_LABEL_FILE_PATH = "./data/train/labels/"
VAL_IMAGE_FILE_PATH = "./data/val/images/"
VAL_LABEL_FILE_PATH = "./data/val/labels/"
CLASSIFIERS_FOLDER_PATH = "./supervised_learning/classifiers/saves/"
WINDOW_SIZES = [(64, 64), (128, 128), (256, 256),(512,512)]  # Window sizes during slidding window process