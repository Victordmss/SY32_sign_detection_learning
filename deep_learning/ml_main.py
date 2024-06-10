from signClassifier import Dataset
import torch
from ml_utils import *
from utils import *

# Creating a dataset object 
dataset = Dataset( 
    image_dir="./data/train/images/", 
    label_dir="./data/train/labels/") 
  
# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
    dataset=dataset, 
    batch_size=1, 
    shuffle=True, 
) 
  
# Getting a batch from the dataloader 
x, y = next(iter(loader)) 
  
# Getting the boxes coordinates and converting them into bounding boxes
bboxes = []
for bbox in y:
    bboxes.append([value.item() for value in bbox])
  
# Plotting the image with the bounding boxes 
plot_bbox_image(x[0].to("cpu"), bboxes)
