import numpy as np
import torch
from torch import tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from ml_utils import *
from utils import *
import csv
from PIL import Image


# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
    def __init__( 
        self, image_dir, label_dir, anchors, 
        image_size=416, grid_sizes=[13, 26, 52]
        , transform=None): 
                    
        # Image and label directories 
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        # Image size 
        self.image_size = image_size 
        # Transformations 
        self.transform = transform
        # Grid sizes for each scale 
        self.grid_sizes = grid_sizes 
        # Anchor boxes 
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # A COMPRENDRE
        # Number of anchor boxes 
        self.num_anchors = self.anchors.shape[0]  # A COMPRENDRE
        # Number of anchor boxes per scale 
        self.num_anchors_per_scale = self.num_anchors // 3 # A COMPRENDRE
        # Number of classes 
        self.num_classes = len(name_to_int)
        # Ignore IoU threshold 
        self.iou_threshold = 0.5

    def __len__(self): 
        return len(os.listdir(self.image_dir))
    
    def __getitem__(self, idx): 
        # Getting the label path 
        label_path = os.path.join(self.label_dir, (str(idx).zfill(4)+".csv"))
        if os.path.exists(label_path):
            # Getting the image path 
            img_path = os.path.join(self.image_dir, (str(idx).zfill(4)+".jpg")) 
            image = np.array(Image.open(img_path).convert("RGB")) 

            # Creating the label array
            # 5 columns: x0, y0, width (currently x1), height (currently y1), class_label (currently as str)
            with open(label_path, "r") as file:
                        reader = csv.reader(file)
                        bboxes = list(reader)            
            
            if not bboxes:
                # Compute empty bbox with "empty" classe
                pass


            # Process changes on bbox definition
            for box in bboxes:
                box[4] = name_to_int.get(box[4])  # Get the class name as int
                box[:] = map(int, box)
                box[2] = box[2] - box[0]         # Compute width
                box[3] = box[3] - box[1]         # Compute height
                
                # Normalise box
                box[0] = box[0] / image.shape[1]
                box[1] = box[1] / image.shape[0]
                box[2] = box[2] / image.shape[1]
                box[3] = box[3] / image.shape[0]

            # Albumentations augmentations 
            if self.transform: 
                augs = self.transform(image=image, bboxes=bboxes) 
                image = augs["image"] 
                bboxes = augs["bboxes"]

            # Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
            # target : [probabilities, x, y, width, height, class_label] 
            targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) for s in self.grid_sizes] 
            # Identify anchor box and cell for each bounding box 
            for box in bboxes: 
                # Calculate iou of bounding box with anchor boxes 
                iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False) 

                # Selecting the best anchor box 
                anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
                x, y, width, height, class_label = box 

                # At each scale, assigning the bounding box to the best matching anchor box 
                has_anchor = [False] * 3
                for anchor_idx in anchor_indices: 
                    scale_idx = anchor_idx // self.num_anchors_per_scale 
                    anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                    
                    # Identifying the grid size for the scale 
                    s = self.grid_sizes[scale_idx] 
                    
                    # Identifying the cell to which the bounding box belongs 
                    i, j = int(s * y), int(s * x) 
                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                    
                    # Check if the anchor box is already assigned 
                    if not anchor_taken and not has_anchor[scale_idx]: 

                        # Set the probability to 1 
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                        # Calculating the center of the bounding box relative 
                        # to the cell 
                        x_cell, y_cell = s * x - j, s * y - i 

                        # Calculating the width and height of the bounding box 
                        # relative to the cell 
                        width_cell, height_cell = (width * s, height * s) 

                        # Idnetify the box coordinates 
                        box_coordinates = torch.tensor( 
                                            [x_cell, y_cell, width_cell, 
                                            height_cell] 
                                        ) 

                        # Assigning the box coordinates to the target 
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 

                        # Assigning the class label to the target 
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

                        # Set the anchor box as assigned for the scale 
                        has_anchor[scale_idx] = True

                    # If the anchor box is already assigned, check if the 
                    # IoU is greater than the threshold 
                    elif not anchor_taken and iou_anchors[anchor_idx] > self.iou_threshold: 
                        # Set the probability to -1 to ignore the anchor box 
                        targets[scale_idx][anchor_on_scale, i, j, 0] = -1
            # Return the image and the target 
            return image, tuple(targets)
        else:
            return

# Defining CNN Block 
class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__() 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 
		self.use_batch_norm = use_batch_norm 

	def forward(self, x): 
		# Applying convolution 
		x = self.conv(x) 
		# Applying BatchNorm and activation if needed 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x


# Defining residual block 
class ResidualBlock(nn.Module): 
	def __init__(self, channels, use_residual=True, num_repeats=1): 
		super().__init__() 
		
		# Defining all the layers in a list and adding them based on number of 
		# repeats mentioned in the design 
		res_layers = [] 
		for _ in range(num_repeats): 
			res_layers += [ 
				nn.Sequential( 
					nn.Conv2d(channels, channels // 2, kernel_size=1), 
					nn.BatchNorm2d(channels // 2), 
					nn.LeakyReLU(0.1), 
					nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
					nn.BatchNorm2d(channels), 
					nn.LeakyReLU(0.1) 
				) 
			] 
		self.layers = nn.ModuleList(res_layers) 
		self.use_residual = use_residual 
		self.num_repeats = num_repeats 
	
	# Defining forward pass 
	def forward(self, x): 
		for layer in self.layers: 
			residual = x 
			x = layer(x) 
			if self.use_residual: 
				x = x + residual 
		return x


# Defining scale prediction class 
class ScalePrediction(nn.Module): 
	def __init__(self, in_channels, num_classes): 
		super().__init__() 
		# Defining the layers in the network 
		self.pred = nn.Sequential( 
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
			nn.BatchNorm2d(2*in_channels), 
			nn.LeakyReLU(0.1), 
			nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1), 
		) 
		self.num_classes = num_classes 
	
	# Defining the forward pass and reshaping the output to the desired output 
	# format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
	def forward(self, x): 
		output = self.pred(x) 
		output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
		output = output.permute(0, 1, 3, 4, 2) 
		return output


# Class for defining YOLOv3 model 
class YOLOv3(nn.Module): 
    def __init__(self, in_channels=3, num_classes=20): 
        super().__init__() 
        self.num_classes = num_classes 
        self.in_channels = in_channels 
  
        # Layers list for YOLOv3 
        self.layers = nn.ModuleList([ 
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(64, num_repeats=1), 
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(128, num_repeats=2), 
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(256, num_repeats=8), 
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(512, num_repeats=8), 
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(1024, num_repeats=4), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(1024, use_residual=False, num_repeats=1), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
            ScalePrediction(512, num_classes=num_classes), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=2), 
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0), 
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(512, use_residual=False, num_repeats=1), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
            ScalePrediction(256, num_classes=num_classes), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=2), 
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0), 
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(256, use_residual=False, num_repeats=1), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
            ScalePrediction(128, num_classes=num_classes) 
        ]) 
      
    # Forward pass for YOLOv3 with route connections and scale predictions 
    def forward(self, x): 
        outputs = [] 
        route_connections = [] 
  
        for layer in self.layers: 
            if isinstance(layer, ScalePrediction): 
                outputs.append(layer(x)) 
                continue
            x = layer(x) 
  
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8: 
                route_connections.append(x) 
              
            elif isinstance(layer, nn.Upsample): 
                x = torch.cat([x, route_connections[-1]], dim=1) 
                route_connections.pop() 
        return outputs
