from tqdm import tqdm
import torch
import torch.nn as nn


# Define the neural network model
class NeuralNet(nn.Module):
    """
    A neural network model with multiple fully connected layers, batch normalization,
    dropout for regularization, and Leaky ReLU activation functions.

    Attributes:
        input_size (int): The size of the input features.
        slope (float): Negative slope for the Leaky ReLU activation function.
        fc1, fc2, ..., fc6 (nn.Linear): Fully connected layers.
        bn1, bn2, ..., bn5 (nn.BatchNorm1d): Batch normalization layers.
        dropout1, dropout2, ..., dropout5 (nn.Dropout): Dropout layers for regularization.
    
    Methods:
        forward(x): Performs the forward pass through the network.
    """
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
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
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
    """
    Trains the neural network model for a specified number of epochs.

    Args:
        num_epochs (int): Number of epochs to train the model.
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function to be optimized.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (optional, torch.optim.lr_scheduler): Learning rate scheduler. Default is None.

    Returns:
        list: A list of average loss values for each epoch.
    """
    loss_list = []
    for _ in tqdm(range(num_epochs), ncols = 100):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for x_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(x_batch).squeeze()
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
