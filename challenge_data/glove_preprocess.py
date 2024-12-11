import os
import re
import emoji
import random
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

# Enable tqdm for pandas
tqdm.pandas()

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


def preprocess_tweet(tweet):
    # Lowercasing
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\.\S+', '', tweet)
    
    # Remove mentions (@username)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtags (keep the text after the #)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Remove special characters, punctuation, and numbers
    tweet = re.sub(r'[^a-z\s]', '', tweet)
    
    # Tokenization
    words = tweet.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Handle emojis (optional: convert to text or remove)
    tweet = emoji.demojize(' '.join(words))  # Converts emojis to text, e.g., ":smile:"
    
    # Final cleanup: remove redundant spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

# Get a sorted list of filenames
file_list = sorted(os.listdir("train_tweets"))

# Randomly select 4 files for df_test
test_files = random.sample(file_list, 4)

# Get the remaining files for df_train
train_files = [file for file in file_list if file not in test_files]

# Combine train files into df_train
li_train = []
for filename in train_files:
    df = pd.read_csv("train_tweets/" + filename)
    li_train.append(df)
df_train = pd.concat(li_train, ignore_index=True)

# Combine test files into df_test
li_test = []
for filename in test_files:
    df = pd.read_csv("train_tweets/" + filename)
    li_test.append(df)
df_test = pd.concat(li_test, ignore_index=True)

# Output df_train and df_test for verification
print(f"df_train: {df_train.shape}")
print(f"df_test: {df_test.shape}")

# Apply preprocessing to each tweet
df_train['Tweet'] = df_train['Tweet'].progress_apply(preprocess_tweet)
df_test['Tweet'] = df_test['Tweet'].progress_apply(preprocess_tweet)

# Add a feature for sentiment using TextBlob
df_train['sentiment'] = df_train['Tweet'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
df_test['sentiment'] = df_test['Tweet'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)

# Save to a csv file
df_train.to_csv('train_data_preprocess.csv')
df_test.to_csv('test_data_preprocess.csv')

# Function to subdivide data into 20 intervals and concatenate embeddings
def create_subdivisions_with_concatenation(df, num_subdivisions=20, embeddings_model=None):
    subdivided_data = []
    
    # Convert Tweet columns in strings to avoid mistakes
    df['Tweet'] = df['Tweet'].astype(str) 

    # Group by MatchID and PeriodID
    for (match_id, period_id), group in df.groupby(['MatchID', 'PeriodID']):
        tweets_per_period = len(group)
        subdivision_size = tweets_per_period // num_subdivisions
        
        # Placeholder for concatenated embeddings
        concatenated_embeddings = []
        
        for i in range(num_subdivisions):
            # Get the subset of tweets for this subdivision
            start_idx = i * subdivision_size
            end_idx = (i + 1) * subdivision_size if (i + 1) * subdivision_size <= tweets_per_period else tweets_per_period
            
            if start_idx >= end_idx:  # Handle edge cases
                continue
            
            subdivision_tweets = group.iloc[start_idx:end_idx]
            
            # Compute average embedding for this subdivision
            embeddings = []
            for tweet in subdivision_tweets['Tweet']:
                try:
                    # Replace this with your actual embedding model logic
                    embeddings.append(get_avg_embedding(tweet, embeddings_model))
                except Exception as e:
                    print(f"Error processing tweet: {tweet} | Error: {e}")
                    continue
            
            if embeddings:
                avg_embedding = np.mean(np.vstack(embeddings), axis=0)
                concatenated_embeddings.append(avg_embedding)
        
        # Flatten concatenated embeddings into a single vector
        if concatenated_embeddings:
            flattened_embeddings = np.concatenate(concatenated_embeddings)
            
            subdivided_data.append({
                'MatchID': match_id,
                'PeriodID': period_id,
                'ID': group['ID'].iloc[0],  # Keep the first ID
                'ConcatenatedEmbeddings': flattened_embeddings,
                'EventType': group['EventType'].iloc[0]  # Assuming same EventType for the whole period
            })
    
    return pd.DataFrame(subdivided_data)

df_train = pd.read_csv('train_data_preprocess.csv')
df_test = pd.read_csv('test_data_preprocess.csv')

df_train_subdivided = create_subdivisions_with_concatenation(df_train, 
                                                             num_subdivisions=20, 
                                                             embeddings_model=embeddings_model)
df_test_subdivided = create_subdivisions_with_concatenation(df_test, 
                                                             num_subdivisions=20, 
                                                             embeddings_model=embeddings_model)

df_train_subdivided.to_pickle('train_subdivided_data.pkl')
df_test_subdivided.to_pickle('test_subdivided_data.pkl')

# Preparing evaluation data: there is no event type column so the function needs to be different

def prepare_eval_data_with_concatenation(eval_df, num_subdivisions=20, embeddings_model=None):
    prepared_data = []
    
    # Convertir la colonne Tweet en chaînes pour éviter les erreurs
    eval_df['Tweet'] = eval_df['Tweet'].astype(str)
    
    for (match_id, period_id), group in eval_df.groupby(['MatchID', 'PeriodID']):
        tweets_per_period = len(group)
        subdivision_size = tweets_per_period // num_subdivisions
        
        concatenated_embeddings = []
        
        for i in range(num_subdivisions):
            # Get the subset of tweets for this subdivision
            start_idx = i * subdivision_size
            end_idx = (i + 1) * subdivision_size if (i + 1) * subdivision_size <= tweets_per_period else tweets_per_period
            
            if start_idx >= end_idx:
                continue
            
            subdivision_tweets = group.iloc[start_idx:end_idx]
            
            # Compute average embedding for this subdivision
            embeddings = []
            for tweet in subdivision_tweets['Tweet']:
                try:
                    embeddings.append(get_avg_embedding(tweet, embeddings_model))
                except Exception as e:
                    print(f"Error processing tweet: {tweet} | Error: {e}")
                    continue
            
            if embeddings:
                avg_embedding = np.mean(np.vstack(embeddings), axis=0)
                concatenated_embeddings.append(avg_embedding)
        
        # Flatten concatenated embeddings into a single vector
        if concatenated_embeddings:
            flattened_embeddings = np.concatenate(concatenated_embeddings)
            
            prepared_data.append({
                'ID': group['ID'].iloc[0],  # Keep the first ID
                'ConcatenatedEmbeddings': flattened_embeddings,
                'PeriodID': period_id
            })
    
    return pd.DataFrame(prepared_data)

li = []
for filename in os.listdir("eval_tweets"):
    df = pd.read_csv("eval_tweets/" + filename)
    li.append(df)
df_eval = pd.concat(li, ignore_index=True)

# Apply preprocessing to each tweet
df_eval['Tweet'] = df_eval['Tweet'].progress_apply(preprocess_tweet)

# Add a feature for sentiment using TextBlob
df_eval['sentiment'] = df_eval['Tweet'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)

# Save to a csv file
df_eval.to_csv('eval_data_preprocess.csv')

df_eval = pd.read_csv('eval_data_preprocess.csv')

df_eval_subdivided = prepare_eval_data_with_concatenation(df_eval, num_subdivisions=20, embeddings_model=embeddings_model)

df_eval_subdivided.to_pickle('eval_subdivided_data.pkl')
