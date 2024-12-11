'''
Pour chaque minute, utiliser un CNN qui apprend la représentation des ~10000 tweets que l'on donne ensuite au MLP
'''
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, slope = 0.01):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.dropout5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(16, 1)
        
        self.slope = slope

    def forward(self, x):
        x = self.bn1(torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=self.slope))
        x = self.dropout1(x)
        x = self.bn2(torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=self.slope))
        x = self.dropout2(x)
        x = self.bn3(torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=self.slope))
        x = self.dropout3(x)
        x = self.bn4(torch.nn.functional.leaky_relu(self.fc4(x), negative_slope=self.slope))
        x = self.dropout4(x)
        x = self.bn5(torch.nn.functional.leaky_relu(self.fc5(x), negative_slope=self.slope))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x

# Training loop with increased epochs
def train(num_epochs, model, train_loader, criterion, optimizer, scheduler=None):
    loss_list = []
    for epoch in tqdm(range(num_epochs), ncols = 100):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch).squeeze()
            y_batch = y_batch.float().squeeze()

            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count +=1
        if scheduler:
            scheduler.step()
        loss_list.append(epoch_loss/batch_count)

    return loss_list
