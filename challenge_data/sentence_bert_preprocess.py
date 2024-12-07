import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from textblob import TextBlob
from sentence_transformers import SentenceTransformer  # Importing Sentence-BERT

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')


# Preprocessing function, can be basic or enhanced
def preprocess_text(text, enhanced=True):
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

    if enhanced:
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
    
    if enhanced:
        # Stemming (optional, after lemmatization)
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    # Join back into a single string
    processed_text = ' '.join(words)
    
    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text


# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose different models

# Read all training files and concatenate them into one dataframe
li = []
for filename in tqdm(os.listdir("train_tweets")):
    df = pd.read_csv("train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)
df = df.sample(n=100000, random_state=42)

# Apply preprocessing to each tweet
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Add a feature for sentiment using TextBlob, after preprocessing
df['sentiment'] = df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Obtain embeddings using Sentence-BERT
df['Embeddings'] = list(model.encode(df['Tweet'].tolist()))  # Directly get embeddings for each tweet

# Remove rows with NaN embedding vectors
df.dropna(inplace=True)

# Drop the columns that are not useful anymore
df = df.drop(columns=['Timestamp', 'Tweet'])

# Drop the columns that are not useful anymore
period_features = df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

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
df_combined.to_csv("sentence_bert_features_and_labels.csv", index=False)
print("Features and labels saved to sentence_bert_features_and_labels.csv")