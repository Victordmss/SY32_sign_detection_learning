import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device 
NB_EPOCHS = 20  # Number of epochs for training 
IMAGE_SIZE = 416  # Image size 
