'''
Pour chaque minute, utiliser un CNN qui apprend la représentation des ~10000 tweets que l'on donne ensuite au MLP
'''
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')


# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


# Basic preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove 'rt' at the beginning of the text 
    text = re.sub(r'^rt\s+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Fonction pour calculer la moyenne et le maximum, puis les concaténer
def compute_mean_and_max_concat(group):
    embeddings = np.stack(group['Embeddings'].values)  # Empilement des vecteurs
    mean_vector = embeddings.mean(axis=0)             # Moyenne des vecteurs
    max_vector = embeddings.max(axis=0)               # Max par coordonnée
    concatenated = np.concatenate((mean_vector, max_vector))  # Concatenation mean + max
    return concatenated


# Define the neural network model with more complexity
class NeuralNet(nn.Module):
    def __init__(self, input_size, slope = 0.01):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.slope = slope

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=self.slope)
        x = self.dropout1(x)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=self.slope)
        x = self.dropout2(x)
        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=self.slope)
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
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
            outputs = model(X_batch)
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
