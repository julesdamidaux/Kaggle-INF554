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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from tqdm import tqdm
tqdm.pandas(desc='Preprocessing')

# Download some NLP models for processing
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

# Entrainer sur un petit échantillon
# df = df.sample(n=1000000, random_state=42)

# Apply preprocessing to each tweet
df['Tweet'] = df['Tweet'].progress_apply(preprocess_text)

# Add a feature for sentiment using TextBlob
df['sentiment'] = df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Apply preprocessing to each tweet and obtain vectors
vector_size = 200  # Adjust based on the chosen GloVe model
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])

# Add tweet_vectors as a new column in the DataFrame
df['tweet_vectors'] = list(tweet_vectors)

# Remove rows with NaN embedding vectors
df.dropna(inplace=True)

# Drop the columns that are not useful anymore
df = df.drop(columns=['Timestamp', 'Tweet'])

# Group the tweets into their corresponding periods
period_features = df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

# Check the length of the DataFrame after processing
print("Number of rows in period_features after processing:", len(period_features))

# Recreate X and y
X = np.hstack([
    np.vstack(period_features['tweet_vectors']),
    period_features['PeriodID'].values.reshape(-1, 1),
    period_features['sentiment'].values.reshape(-1, 1)
])
y = period_features['EventType'].values

# Ensure X and y have the same length
print("Length of X:", len(X))
print("Length of y:", len(y))

###### Evaluating on a test set:

# We split our data into a training and test set that we can use to train our classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# We set up a basic classifier using an MLP (Multi-Layer Perceptron)
clf = MLPClassifier(hidden_layer_sizes=(200, 100, 50, 25), activation='relu', random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

# Calculate the accuracy on our test set
y_pred = clf.predict(X_test)
print("Test set accuracy: ", accuracy_score(y_test, y_pred))


################################## KAGGLE ###########################################

# Lire les fichiers CSV dans eval_tweets et concaténer dans un DataFrame
eval_data = []
for filename in os.listdir("eval_tweets"):
    df = pd.read_csv(os.path.join("eval_tweets", filename))
    eval_data.append(df)
eval_df = pd.concat(eval_data, ignore_index=True)

# Appliquer le prétraitement et ajouter des caractéristiques
eval_df['Tweet'] = eval_df['Tweet'].progress_apply(preprocess_text)
eval_df['sentiment'] = eval_df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
vector_size = 200
eval_tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in eval_df['Tweet']])

# Recréer les caractéristiques X
X_eval = np.hstack([
    eval_tweet_vectors,
    eval_df['PeriodID'].values.reshape(-1, 1),
    eval_df['sentiment'].values.reshape(-1, 1)
])

# Effectuer des prédictions
eval_df['PredictedEventType'] = clf.predict(X_eval)

# Sauvegarder les prédictions dans un fichier CSV
eval_df.to_csv('eval_predictions.csv', index=False)

print("Les prédictions ont été sauvegardées dans 'eval_predictions.csv'.")
