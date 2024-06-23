import torch
from torch.utils.data import Dataset, DataLoader
from deep_learning.config import *
from deep_learning.utils import *
from utils.utils import *
from torch import tensor
from skimage.util import img_as_float32
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.transform import resize

# Create a dataset class to load the images and labels from the folder 
class ClassificationDataset(Dataset): 
    def __init__(self, image_dir, label_dir):         
        # Image and label directories 
        self.X, self.y = datas_to_XY_dataset(image_dir, label_dir)
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        self.transform = TRANFORMATIONS


    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx): 
        x = img_as_float32(self.transform(Image.fromarray(self.X[idx])))
        x.resize((3, 64, 64))
        x = tensor(x)
        y = [np.float32(self.y[idx])]
        y = tensor(y)      
        return x, y
    
    def visualize_repartition(self):   
        plt.hist(self.y, bins=np.arange(self.y.min() - 0.5, self.y.max()+1.5, 1), edgecolor='black')  
        plt.xlabel('Valeurs')
        plt.ylabel('Fréquence')
        plt.title('Histogramme de répartition des classes')
        plt.show()


# Create a classification & extraction features network
class SignNet(nn.Module):
    def __init__(self, num_classes):
        super(SignNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = np.array(x)
        with torch.no_grad():
            # Convert numpy array to tensor
            x = torch.from_numpy(x).float()
            out = self.conv_layer1(x)
            out = self.conv_layer2(out)
            out = self.max_pool1(out)
        
            out = self.conv_layer3(out)
            out = self.conv_layer4(out)
            out = self.max_pool2(out)
                    
            out = out.view(out.size(0), -1)
        
            features = self.fc1(out)
        return features

# Version 1 is for the first part of the development, extraction & classification from the same network
class SignClassifierV1():
    def __init__(self):
        self.net = SignNet(NB_CLASSES)
        self.criterion = torch.nn.CrossEntropyLoss()  
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = 0.001, weight_decay = 0.005, momentum = 0.9)  
        self.training_dataset = ClassificationDataset(TRAINING_IMAGE_FILE_PATH, TRAINING_LABEL_FILE_PATH)
    
    def fit(self):
        loader = DataLoader(self.training_dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_courant = 0.0
        nb_correct = 0.0
        nb_courant = 0.0
        PRINT_INTERVAL = 50
        for epoch in range(NB_EPOCHS):
            #Load in the data in batches using the train_loader object
            for i, (images, labels) in enumerate(loader):  
                # Move tensors to the configured device
                images = images
                labels = labels
                
                # Forward pass
                outputs = self.net(images)
                loss = self.criterion(outputs, labels.squeeze().long())
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                nb_correct += (predicted == labels.squeeze().long()).sum().item()
                nb_courant += BATCH_SIZE
                train_err = 1 - nb_correct / nb_courant
            print('Epoch [{}/{}], Loss: {:.4f}, Train Error: {:.4f}'.format(epoch+1, NB_EPOCHS, loss.item(), train_err))           

        return self

    def evaluate_classification(self):
        validation_dataset = ClassificationDataset(VAL_IMAGE_FILE_PATH, VAL_LABEL_FILE_PATH)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
        correct = 0
        total = 0
        for images, labels in validation_loader:
            labels = labels.squeeze().long()
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the classification network : {100 * correct / total} %')

    def predict(self, X, visualize=False):
        predictions = []
      
        # Perform selective search to compute ROIs (Regions of interest)
        rois = selective_search(X)[:500]

        image = Image.fromarray(X)

        # Step 2: Classification of the propositions
        scored_bboxes = []
        
        for (x, y, w, h) in rois:
            roi = image.crop((x, y, x + w, y + h))
            roi = TRANFORMATIONS(roi).unsqueeze(0)
            output = self.net(roi)
            probabilities = F.softmax(output, dim=1)
            probability, predicted = torch.max(probabilities.data, 1)
            # Append bounding box and predicted class


            if probability>0.8:
                scored_bboxes.append((x, y, x + w, y + h, predicted.item(), probability.item()))

        scored_bboxes = non_max_suppression(scored_bboxes)

        # Step 3: Post processing of the classification to compute better prediction
        # Apply non-maximum suppression (NMS) to remove overlapping bounding boxes
        
        """proposed_bboxes = nms(scored_bboxes, threshold=0.1)
        print(proposed_bboxes.shape)"""
        
        # Step 4 : Visualize if asked
        if visualize:
            plot_bbox_image(X, scored_bboxes)

        return predictions

# Version 2 is for the second part of the development, extraction from the network and classification by SVM
class SignClassifierV2():
    def __init__(self):
        self.net = SignNet(NB_CLASSES)
        self.training_dataset = ClassificationDataset(TRAINING_IMAGE_FILE_PATH, TRAINING_LABEL_FILE_PATH)
        self.svm_clf = None  # SVM classifier
    
    def fit(self):
        loader = DataLoader(self.training_dataset, batch_size=BATCH_SIZE, shuffle=True)
        feature_list = []
        label_list = []
        
        self.net.eval()  # Set the network to evaluation mode for feature extraction
        
        for i, (images, labels) in enumerate(loader):  
            images = np.array(images)
            
            features = self.net.extract_features(images)
            feature_list.append(features.numpy())
            label_list.append(labels.numpy())
            
        # Convert lists to numpy arrays
        feature_array = np.vstack(feature_list)
        label_array = np.concatenate(label_list)
        
        # Train SVM
        self.svm_clf = svm.SVC(kernel="poly", probability=True)
        self.svm_clf.fit(feature_array, label_array.squeeze())

        return self

    def predict(self, images):
        resized_images = []
        for image in images:
            resized_image = resize(image, (3, 64, 64))  # Resize each image to 3x64x64
            resized_images.append(resized_image)

        # Extract features
        features = self.net.extract_features(np.array(resized_images))
        print(features.shape)

        # Predict using SVM classifier
        predictions = self.svm_clf.predict_proba(features.numpy())

        return predictions
    

    def evaluate_classification(self):
        validation_dataset = ClassificationDataset(TRAINING_IMAGE_FILE_PATH, TRAINING_LABEL_FILE_PATH)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
        correct = 0
        total = 0

        feature_list = []
        label_list = []
        
        self.net.eval()  # Set the network to evaluation mode for feature extraction
        
        for i, (images, labels) in enumerate(validation_loader):  
            images = np.array(images)
            
            features = self.net.extract_features(images)
            feature_list.append(features.numpy())
            label_list.append(labels.numpy())
            
        # Convert lists to numpy arrays
        feature_array = np.vstack(feature_list)
        label_array = np.concatenate(label_list)
        
        # Train SVM
        y = self.svm_clf.predict(feature_array)
        
        print(f'Accuracy of the classification network : {np.mean(y == label_array)} %')