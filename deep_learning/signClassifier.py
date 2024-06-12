import torch
from torch.utils.data import Dataset, DataLoader
from deep_learning.config import *
from utils.utils import *
from torch import tensor
from skimage.util import img_as_float32
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# Create a dataset class to load the images and labels from the folder 
class SignDataset(Dataset): 
    def __init__(self, image_dir, label_dir):         
        # Image and label directories 
        self.X, self.y = load_dataset(image_dir, label_dir)
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        self.transform = TRANFORMATIONS


    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx): 
        x = img_as_float32(self.transform(Image.fromarray(self.X[idx])))
        x.resize((3, 32, 32))
        x = tensor(x)
        y = [np.float32(self.y[idx])]
        y = tensor(y)      
        return x, y
    
    def visualize_repartition(self):    
        plt.hist(self.y, bins=np.arange(self.y.min(), self.y.max()+1)-0.5, edgecolor='black')  
        plt.xlabel('Valeurs')
        plt.ylabel('Fréquence')
        plt.title('Histogramme de répartition des classes')
        plt.show()


# Create a classification network
class SignNet(nn.Module):
    def __init__(self, num_classes):
        super(SignNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class SignClassifier():
    def __init__(self):
        self.net = SignNet(NB_CLASSES)
        self.criterion = torch.nn.CrossEntropyLoss()  
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = 0.001, weight_decay = 0.005, momentum = 0.9)  
    
    def fit(self):
        dataset = SignDataset(TRAINING_IMAGE_FILE_PATH, TRAINING_LABEL_FILE_PATH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
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

    def predict(self, loader):
        correct = 0
        total = 0
        for images, labels in loader:
            labels = labels.squeeze().long()
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network : {100 * correct / total} %')

