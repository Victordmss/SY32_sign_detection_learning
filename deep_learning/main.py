from deep_learning.signClassifier import SignClassifier, SignDataset
from torch.utils.data import DataLoader
from deep_learning.config import *
from utils.utils import *
import torchvision
import torchvision.transforms as transforms

# Create & train
clf = SignClassifier()
clf.fit()

# Predict & test
dataset = SignDataset(VAL_IMAGE_FILE_PATH, VAL_LABEL_FILE_PATH)
validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
clf.predict(validation_loader)



