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
for filename in os.listdir("train_tweets"):
    df = pd.read_csv("train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)
df = df.sample(n=10000, random_state=42)

# Apply preprocessing to each tweet
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Apply preprocessing to each tweet and obtain vectors
vector_size = 200  # Adjust based on the chosen GloVe model
df['Embeddings'] = list(np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']]))

# Drop the columns that are not useful anymore
period_features = df.drop(columns=['Timestamp', 'Tweet'])
# Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

# We drop the non-numerical features and keep the embeddings values for each period
X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
X = [x[0].tolist() for x in X]
# We extract the labels of our training samples
y = period_features['EventType'].values

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
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 1)
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
model = NeuralNet(vector_size)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Smaller learning rate

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