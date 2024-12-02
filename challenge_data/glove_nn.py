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
from nltk.stem import WordNetLemmatizer, PorterStemmer
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


# Preprocessing function, can be basic or enhanced
def preprocess_text(text, enhanced = True):
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

    if enhanced :
        contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will", "'t": " not",
        "'ve": " have", "'m": " am"}
        for contraction, full_form in contractions.items():
            text = re.sub(contraction, full_form, text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Remove very short words
        words = [word for word in words if len(word) > 2]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    if enhanced :
        # Stemming (optional, after lemmatization)
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    # Join back into a single string
    processed_text = ' '.join(words)
    
    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text


# Fonction pour calculer la moyenne et le maximum, puis les concaténer
def compute_mean_and_max_concat(group):
    embeddings = np.stack(group['Embeddings'].values)  # Empilement des vecteurs
    mean_vector = embeddings.mean(axis=0)             # Moyenne des vecteurs
    max_vector = embeddings.max(axis=0)               # Max par coordonnée
    concatenated = np.concatenate((mean_vector, max_vector))  # Concatenation mean + max
    return concatenated


# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, slope = 0.01):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(32, 1)
        self.slope = slope

    def forward(self, x):
        x = self.bn1(torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=self.slope))
        x = self.dropout1(x)
        x = self.bn2(torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=self.slope))
        x = self.dropout2(x)
        x = self.bn3(torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=self.slope))
        x = self.dropout3(x)
        x = self.fc4(x)
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
