import torch 
import albumentations as A 
import cv2
from albumentations.pytorch import ToTensorV2 

# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

# Batch size for training 
batch_size = 32

# Learning rate for training 
leanring_rate = 1e-5

# Number of epochs for training 
epochs = 20

# Image size 
image_size = 416

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 

# Class labels 
class_labels = [ 
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# Function to save checkpoint 
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
	print("==> Saving checkpoint") 
	checkpoint = { 
		"state_dict": model.state_dict(), 
		"optimizer": optimizer.state_dict(), 
	} 
	torch.save(checkpoint, filename)


# Function to load checkpoint 
def load_checkpoint(checkpoint_file, model, optimizer, lr): 
	print("==> Loading checkpoint") 
	checkpoint = torch.load(checkpoint_file, map_location=device) 
	model.load_state_dict(checkpoint["state_dict"]) 
	optimizer.load_state_dict(checkpoint["optimizer"]) 

	for param_group in optimizer.param_groups: 
		param_group["lr"] = lr 


# Transform for testing 
test_transform = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=image_size), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        ),

        # Normalize the image 
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ], 
    # Augmentation for bounding boxes  
    bbox_params=A.BboxParams( 
                    format="yolo",  
                    min_visibility=0.4,  
                    label_fields=[] 
                ) 
)
