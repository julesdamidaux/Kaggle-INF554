'''
Pour chaque minute, utiliser un CNN qui apprend la représentation des ~10000 tweets que l'on donne ensuite au MLP
'''



import time 

debut = time.time()

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
# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings


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


# Read all training files and concatenate them into one dataframe
li = []
for filename in tqdm(os.listdir("train_tweets")):
    df = pd.read_csv("train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)
df = df.sample(n=500000, random_state=42)

# Apply preprocessing to each tweet
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Apply preprocessing to each tweet and obtain vectors
vector_size = 200  # Adjust based on the chosen GloVe model
df['Embeddings'] = list(np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']]))

# Drop the columns that are not useful anymore
period_features = df.drop(columns=['Timestamp', 'Tweet'])

# Assurez-vous de remplacer 'Embeddings' par le nom réel de la colonne contenant vos embeddings
grouped = period_features.groupby(['MatchID', 'PeriodID', 'ID'])

# Fonction pour calculer la moyenne et le maximum, puis les concaténer
def compute_mean_and_max_concat(group):
    embeddings = np.stack(group['Embeddings'].values)  # Empilement des vecteurs
    mean_vector = embeddings.mean(axis=0)             # Moyenne des vecteurs
    max_vector = embeddings.max(axis=0)               # Max par coordonnée
    concatenated = np.concatenate((mean_vector, max_vector))  # Concatenation mean + max
    return concatenated

# Application de la fonction sur chaque groupe pour obtenir une liste de listes
X = grouped.apply(lambda g: compute_mean_and_max_concat(g)).tolist()

y = grouped['EventType'].first().tolist()

###### Evaluating on a test set:

# We split our data into a training and test set that we can use to train our classifier without fine-tuning into the
# validation set and without submitting too many times into Kaggle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create a dataset and data loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model with more complexity
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNet(vector_size*2)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Smaller learning rate (combien mettre ??)

# Training loop with increased epochs
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation on the test set
model.eval()
with torch.no_grad():
    y_pred_list = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_pred_list.extend(outputs.numpy())

y_pred = (np.array(y_pred_list) > 0.5).astype(int)  # Adjust threshold for binary classification

# Print test set accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

######################################### KAGGLE SUBMISSION ################################################


# We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
# Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
# to be submitted on Kaggle.
lis = []
for fname in os.listdir("eval_tweets"):
    df = pd.read_csv("eval_tweets/" + fname)
    lis.append(df)

eval_df = pd.concat(lis, ignore_index=True)

eval_df['Tweet'] = eval_df['Tweet'].apply(preprocess_text)

eval_df['Embeddings'] = list(np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in eval_df['Tweet']]))
period_features = eval_df.drop(columns=['Timestamp', 'Tweet'])
grouped = period_features.groupby(['MatchID', 'PeriodID', 'ID'])

# Application de la fonction sur chaque groupe pour obtenir une liste de listes
X_test = grouped.apply(lambda g: compute_mean_and_max_concat(g)).tolist()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Predictions on the eval set
model.eval()
with torch.no_grad():
    y_pred_list = []
    for X_batch in test_loader:
        # Si X_batch est déjà un tenseur, pas besoin de le transformer
        X_batch = X_batch[0]
        
        outputs = model(X_batch)  # Passez un tenseur au modèle
        print(outputs)
        y_pred_list.extend(outputs.numpy())  # Ajoutez les prédictions à la liste


y_pred = (np.array(y_pred_list) > 0.5).astype(int)  # Adjust threshold for binary classification
y_pred = [_[0] for _ in y_pred]
print(y_pred)

IDs = grouped['ID'].first().tolist()
print(IDs)

results = pd.DataFrame({'ID': IDs, 'EventType': y_pred})

results.to_csv('submission.csv', index=False)

fin = time.time()

print('temps total', fin-debut)