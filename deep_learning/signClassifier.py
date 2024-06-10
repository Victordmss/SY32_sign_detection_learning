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

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx): 
        x = img_as_float32(self.X[idx])
        x.resize((1, 24, 24))
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
    def __init__(self):
        super(SignNet, self).__init__()
        # Convolution 2D avec 2 noyaus 3x3 et un padding de 1
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        # Convolution 2D avec 4 noyaux 3x3 et un padding de 1
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        # Couche fully connected avec une seule sortie
        self.fc1 = nn.Linear(144, 1)

    def forward(self, x):
        # Convolution de x avec la couche conv1
        x = self.conv1(x)
        # Fonction d'activation ReLU
        x = F.relu(x)
        # Max pooling sur une fenêtre 2x2
        x = F.max_pool2d(x, (2, 2))
        # Convolution de x avec la couche conv2
        x = self.conv2(x)
        # Fonction d'activation ReLU
        x = F.relu(x)
        # Max pooling sur une fenêtre 2x2
        x = F.max_pool2d(x, (2, 2))
        # Redimensionnement de x en une colonne
        x = x.view(-1, self.fc1.in_features)
        # Opération avec la couche fully connected fc1
        x = self.fc1(x)

        return x


class SignClassifier():
    def __init__(self):
        self.net = SignNet()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)

    def fit(self, n_epoch=NB_EPOCHS, batch_size=BATCH_SIZE):
        dataset = SignDataset(TRAINING_IMAGE_FILE_PATH, TRAINING_LABEL_FILE_PATH)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_courant = 0.0
        nb_correct = 0.0
        nb_courant = 0.0
        PRINT_INTERVAL = 50
        for epoch in range(n_epoch):
            for batch_idx, (X, y) in enumerate(loader):
                # Initialise le gradient courant à zéro
                self.optimizer.zero_grad()
                # Prédit la classe des éléments X du batch courant
                y_pred = self.net(X)
                # Calcul la fonction de coût sur ce batch
                loss = self.criterion(y_pred, y)
                # Rétropropagation pour calculer le gradient
                loss.backward()
                # Effectue un pas de la descente de gradient
                self.optimizer.step()

                loss_courant += loss.item()
                prediction = 2 * (y_pred > 0) - 1
                nb_correct += (prediction == y).sum().item()
                nb_courant += batch_size
                if (batch_idx % PRINT_INTERVAL) == 0 and batch_idx > 0:
                    train_err = 1 - nb_correct / nb_courant
                    print(
                        f'Epoch: {epoch+1:02d} ; '
                        f'Batch: {batch_idx-PRINT_INTERVAL:03d}-{batch_idx:03d} ; '
                        f'train-loss: {loss_courant:.2f} ; '
                        f'train-err: {train_err*100:.2f}'
                    )
                    loss_courant = 0.0
                    nb_correct = 0.0
                    nb_courant = 0.0

        return self

    def predict(self, X):
        X = img_as_float32(X)
        if len(X.shape) < 3:  # Une seule image
            X.resize((1, 1, 24, 24))
        else:
            X = X.reshape((X.shape[0], 1, 24, 24))
        y = self.net(tensor(X))
        y_pred = 2 * (y.detach().numpy().flatten() > 0) - 1
        return y_pred
