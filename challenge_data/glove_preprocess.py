import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from textblob import TextBlob

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


# Function to calculate mean and max and then to concatenate them
def compute_mean_and_max_concat(group):
    embeddings = np.stack(group['Embeddings'].values)  
    mean_vector = embeddings.mean(axis=0)             
    max_vector = embeddings.max(axis=0)               
    concatenated = np.concatenate((mean_vector, max_vector))  
    return concatenated

# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings

# Read all training files and concatenate them into one dataframe
li = []
for filename in tqdm(os.listdir("train_tweets")):
    df = pd.read_csv("train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)
df = df.sample(n=100000, random_state=42)

# Apply preprocessing to each tweet
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Add a feature for sentiment using TextBlob, after preprocesing
df['sentiment'] = df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Apply preprocessing to each tweet and obtain vectors
vector_size = 200  # Adjust based on the chosen GloVe model

df['Embeddings'] = list(np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']]))

# Remove rows with NaN embedding vectors
df.dropna(inplace=True)

# Drop the columns that are not useful anymore
df = df.drop(columns=['Timestamp', 'Tweet'])

# Assign "PackID" to group every 50 rows
df['PackID'] = (df.index // 50)

#  Group the tweets into their corresponding periods
period_features = df.groupby(['MatchID', 'PeriodID', 'ID', 'PackID']).mean().reset_index()

# Recreate X and y
X = np.hstack([
    np.vstack(period_features['Embeddings']),
    period_features['PeriodID'].values.reshape(-1, 1),
    period_features['sentiment'].values.reshape(-1, 1)
])
y = period_features['EventType'].values

# Ensure X and y have the same length
print("Length of X:", len(X))
print("Length of y:", len(y))

# Combine X and y into a DataFrame
feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
df_combined = pd.DataFrame(X, columns=feature_columns)
df_combined['label'] = y

# Save to a CSV file
df_combined.to_csv("glove_features_and_labels.csv", index=False)
print("Features and labels saved to glove_features_and_labels.csv")
