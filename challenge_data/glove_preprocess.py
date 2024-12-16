import pandas as pd
import tqdm
import gensim.downloader as api
import os
import random
from textblob import TextBlob
from tools import preprocess_tweet, create_subdivisions_with_concatenation, prepare_eval_data_with_concatenation

# Enable tqdm for pandas
tqdm.pandas()

# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings


## Train and test preprocessing

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


## Eval preprocessing

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

df_eval_subdivided = prepare_eval_data_with_concatenation(df_eval, 
                                                          num_subdivisions=20, 
                                                          embeddings_model=embeddings_model)

df_eval_subdivided.to_pickle('eval_subdivided_data.pkl')
