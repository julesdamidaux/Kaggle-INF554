import time
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

debut = time.time()

# Charger le tokenizer et le modèle BERT
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Fonction pour extraire les embeddings BERT
def get_bert_embeddings(tweet, model, tokenizer, max_length=50):
    # Tokenizer le tweet
    inputs = tokenizer(tweet, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    # Obtenir les embeddings de BERT
    with torch.no_grad():
        outputs = model(**inputs)
    # Utiliser les embeddings [CLS] (premier token) comme représentation du tweet
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = text.lower()  # Minuscule
    text = re.sub(r'[^\w\s]', '', text)  # Retirer la ponctuation
    text = re.sub(r'\d+', '', text)  # Retirer les chiffres
    text = re.sub(r'^rt\s+', '', text)  # Retirer "rt" au début
    return text.strip()

# Charger et prétraiter les données
li = []
for filename in tqdm(os.listdir("train_tweets")):
    df = pd.read_csv("train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)
df = df.sample(n=500000, random_state=42)
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Extraire les embeddings BERT pour chaque tweet
df['Embeddings'] = df['Tweet'].apply(lambda tweet: get_bert_embeddings(tweet, bert_model, tokenizer))

# Préparer les données pour CNN
def group_embeddings_by_minute(group):
    embeddings = np.stack(group['Embeddings'].values)  # Empiler les vecteurs
    return embeddings

# Grouper les données par minute et préparer les labels
grouped = df.groupby(['MatchID', 'PeriodID', 'ID'])
X = grouped.apply(lambda g: group_embeddings_by_minute(g)).tolist()
y = grouped['EventType'].first().tolist()

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fonction de padding
def pad_sequences(sequences, max_len, vector_size):
    """Pad sequences to a fixed length."""
    padded = np.zeros((len(sequences), max_len, vector_size))
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, :] = seq[:length]
    return padded

# Paramètres pour le padding
max_tweets = 50
vector_size = bert_model.config.hidden_size  # Taille des embeddings BERT

# Appliquer le padding
X_train_padded = pad_sequences(X_train, max_tweets, vector_size)
X_test_padded = pad_sequences(X_test, max_tweets, vector_size)

# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Création des datasets et loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Définition du modèle CNN
class CNNModel(nn.Module):
    def __init__(self, vector_size, max_tweets):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=vector_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (max_tweets // 2), 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Changer les dimensions pour Conv1D
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Aplatir
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = CNNModel(vector_size=vector_size, max_tweets=max_tweets)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Boucle d'entraînement
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Évaluation sur l'ensemble de test
model.eval()
with torch.no_grad():
    y_pred_list = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_pred_list.extend(outputs.numpy())

y_pred = (np.array(y_pred_list) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

fin = time.time()
print('Temps total:', fin - debut)
