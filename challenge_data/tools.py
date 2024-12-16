import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
import emoji
from nltk.corpus import stopwords


# Download some NLP models for processing
nltk.download('stopwords')
nltk.download('wordnet')


def get_avg_embedding(tweet, model, vector_size=200):
    """
    Computes the average word vector for a given tweet using a pre-trained word embedding model.

    Parameters:
    ----------
    tweet : str
        The input tweet as a string of text.
    model : gensim.models.KeyedVectors or similar
        A pre-trained word embedding model (e.g., Word2Vec, FastText, GloVe) that provides 
        vector representations for words.
    vector_size : int, optional (default=200)
        The dimensionality of the word vectors in the model.

    Returns:
    -------
    numpy.ndarray
        A vector of shape `(vector_size,)` representing the average embedding of the words 
        in the tweet.
        If none of the words in the tweet exist in the model's vocabulary, a zero vector of 
        shape `(vector_size,)` is returned.

    Notes:
    -----
    - The tweet is tokenized by splitting it on whitespace.
    - Words not present in the model's vocabulary are ignored in the computation.
    - If no valid words are found in the tweet, the function returns a zero vector.
    """

    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def preprocess_tweet(tweet):
    """
    Preprocesses a tweet by cleaning and normalizing the text.

    Steps:
    ------
    1. Converts text to lowercase.
    2. Removes URLs, mentions, special characters, and numbers.
    3. Extracts text from hashtags.
    4. Tokenizes the text and removes stopwords.
    5. Applies lemmatization to reduce words to their base form.
    6. Converts emojis to descriptive text (e.g., ":smile:").
    7. Cleans up extra spaces.

    Returns:
    --------
    str
        The cleaned and preprocessed tweet.
    """
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


def create_subdivisions_with_concatenation(df, num_subdivisions=20, embeddings_model=None):
    """
    Divides match data into subdivisions, computes average tweet embeddings for each subdivision, 
    and concatenates them into a single feature vector.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input data containing tweets with columns 'MatchID', 'PeriodID', 'Tweet', 'ID', 
        and 'EventType'.
    num_subdivisions : int, optional (default=20)
        The number of subdivisions to split each match period into.
    embeddings_model : object, optional
        A pre-trained embedding model used to compute tweet embeddings.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with concatenated embeddings for each match period and additional metadata.
    """

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

            if (i + 1) * subdivision_size <= tweets_per_period :
                end_idx = (i + 1) * subdivision_size
            else :
                end_idx = tweets_per_period

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
                'EventType': group['EventType'].iloc[0]  
            })

    return pd.DataFrame(subdivided_data)


def prepare_eval_data_with_concatenation(eval_df, num_subdivisions=20, embeddings_model=None):
    """
    Processes evaluation data by splitting match periods into subdivisions, 
    computing average tweet embeddings, and concatenating them into feature vectors.

    Parameters:
    ----------
    eval_df : pandas.DataFrame
        DataFrame with columns like 'MatchID', 'PeriodID', 'Tweet', and 'ID'.
    num_subdivisions : int, optional
        Number of subdivisions per match period (default=20).
    embeddings_model : object, optional
        Pre-trained model for generating tweet embeddings.

    Returns:
    -------
    pandas.DataFrame
        DataFrame containing concatenated embeddings and metadata ('ID' and 'PeriodID').
    """

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

            if (i + 1) * subdivision_size <= tweets_per_period :
                end_idx = (i + 1) * subdivision_size
            else :
                end_idx = tweets_per_period

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
