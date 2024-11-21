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
df = df.sample(n=10000, random_state=42)

# Apply preprocessing to each tweet
df['Tweet'] = df['Tweet'].apply(preprocess_text)

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

# Recreate X and y after dropping NaN values
X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
X = [x[0].tolist() for x in X]
y = period_features['EventType'].values

print(X[0]) # cest un array : bizarre...
# Ensure X and y have the same length
print("Length of X:", len(X))
print("Length of y:", len(y))


###### Evaluating on a test set:

# We split our data into a training and test set that we can use to train our classifier without fine-tuning into the
# validation set and without submitting too many times into Kaggle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# We set up a basic classifier that we train and then calculate the accuracy on our test set
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

###### For Kaggle submission


# This time we train our classifier on the full dataset that it is available to us.
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)

# We add a dummy classifier for sanity purposes
dummy_clf = DummyClassifier(strategy="most_frequent").fit(X, y)
predictions = []
dummy_predictions = []

# We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
# Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
# to be submitted on Kaggle.

for fname in os.listdir("eval_tweets"):
    val_df = pd.read_csv("eval_tweets/" + fname)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values
    preds = clf.predict(X)
    dummy_preds = dummy_clf.predict(X)

    period_features['EventType'] = preds
    period_features['DummyEventType'] = dummy_preds
    
    predictions.append(period_features[['ID', 'EventType']])
    dummy_predictions.append(period_features[['ID', 'DummyEventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('logistic_predictions.csv', index=False)
pred_df = pd.concat(dummy_predictions)
pred_df.to_csv('dummy_predictions.csv', index=False)

fin = time.time()

print('temps total', fin - debut)