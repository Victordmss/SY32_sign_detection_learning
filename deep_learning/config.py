import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device 
NB_EPOCHS = 3  # Number of epochs for training 
BATCH_SIZE = 5  # Size of the training batch
IMAGE_SIZE = 416  # Image size 
TRAINING_IMAGE_FILE_PATH = "./data/train/images/"
TRAINING_LABEL_FILE_PATH = "./data/train/labels/"
VAL_IMAGE_FILE_PATH = "./data/val/images/"
VAL_LABEL_FILE_PATH = "./data/val/labels/"