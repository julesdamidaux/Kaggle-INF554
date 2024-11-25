import os
import re
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BigBirdTokenizer,
    BigBirdForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
tqdm.pandas(desc='Preprocessing !')
import nltk

torch.cuda.empty_cache()

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Advanced text preprocessing
def preprocess_text(text, remove_stopwords=True):
    """
    Preprocess a given text: normalize, clean, tokenize, lemmatize, and optionally remove stopwords.
    
    Parameters:
        text (str): The input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords (default: True).
        
    Returns:
        str: The cleaned and preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Expand common contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will", "'t": " not",
        "'ve": " have", "'m": " am"
    }
    for contraction, full_form in contractions.items():
        text = re.sub(contraction, full_form, text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)  # Punctuation
    text = re.sub(r'\d+', '', text)      # Digits
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    # Remove very short words
    words = [word for word in words if len(word) > 2]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Stemming (optional, after lemmatization)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Join back into a single string
    processed_text = ' '.join(words)
    
    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

# Load and preprocess data
def load_and_preprocess_data(folder_path):
    data = []
    print("Loading and preprocessing data...")
    for filename in tqdm(os.listdir(folder_path), desc="Files processed"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df = df.sample(n=100000, random_state=42)
        df['Tweet'] = df['Tweet'].progress_apply(preprocess_text)
        data.append(df)
    return pd.concat(data, ignore_index=True)

def compute_metrics(eval_pred):
    predictions, labels, sample_ids = eval_pred
    preds = np.argmax(predictions, axis=1)

    # Agrégation des prédictions par sample_id
    df = pd.DataFrame({'sample_id': sample_ids, 'preds': preds, 'labels': labels})
    aggregated = df.groupby('sample_id').agg({
        'preds': lambda x: np.argmax(np.bincount(x)),
        'labels': 'first'
    }).reset_index()

    acc = accuracy_score(aggregated['labels'], aggregated['preds'])
    return {"accuracy": acc}

def main():
    # Load training data
    train_folder = "train_tweets"
    df = load_and_preprocess_data(train_folder)

    # Group tweets by minute
    print("Grouping data by minute...")
    df_grouped = df.groupby(['MatchID', 'PeriodID', 'ID'], as_index=False).agg({
        'Tweet': ' '.join,  # Concatenate tweets
        'EventType': 'first'  # Take first EventType
    })

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(df_grouped, test_size=0.2, random_state=42)

    # Load tokenizer and BigBird model
    print("Loading tokenizer and BigBird model...")
    bigbird_model_name = "google/bigbird-roberta-base"
    tokenizer = BigBirdTokenizer.from_pretrained(bigbird_model_name)
    model = BigBirdForSequenceClassification.from_pretrained(bigbird_model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set max_length and stride for sliding window
    max_length = 1024
    stride = 512

    # Process training data with sliding window
    print("Processing training data with sliding window...")
    train_encodings = {'input_ids': [], 'attention_mask': []}
    train_labels = []

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Processing training data'):
        text = row['Tweet']
        label = row['EventType']
        encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding='max_length',
            return_attention_mask=True,
        )
        num_chunks = len(encoding['input_ids']) if isinstance(encoding['input_ids'][0], list) else 1

        if num_chunks == 1:
            train_encodings['input_ids'].append(encoding['input_ids'])
            train_encodings['attention_mask'].append(encoding['attention_mask'])
            train_labels.append(label)
        else:
            train_encodings['input_ids'].extend(encoding['input_ids'])
            train_encodings['attention_mask'].extend(encoding['attention_mask'])
            overflow_mapping = encoding['overflow_to_sample_mapping']
            train_labels.extend([label] * len(overflow_mapping))

    # Process validation data with sliding window
    print("Processing validation data with sliding window...")
    val_encodings = {'input_ids': [], 'attention_mask': []}
    val_labels = []

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc='Processing validation data'):
        text = row['Tweet']
        label = row['EventType']
        encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding='max_length',
            return_attention_mask=True,
        )
        num_chunks = len(encoding['input_ids']) if isinstance(encoding['input_ids'][0], list) else 1

        if num_chunks == 1:
            val_encodings['input_ids'].append(encoding['input_ids'])
            val_encodings['attention_mask'].append(encoding['attention_mask'])
            val_labels.append(label)
        else:
            val_encodings['input_ids'].extend(encoding['input_ids'])
            val_encodings['attention_mask'].extend(encoding['attention_mask'])
            overflow_mapping = encoding['overflow_to_sample_mapping']
            val_labels.extend([label] * len(overflow_mapping))

    # Create datasets
    class TweetDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    # Configure training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=10,
        fp16=True
    )

    # Define Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Save model
    print("Saving the fine-tuned model...")
    model.save_pretrained("./fine_tuned_bigbird")
    tokenizer.save_pretrained("./fine_tuned_bigbird")

    # Evaluate model
    print("Evaluating the model...")
    raw_pred, labels, _ = trainer.predict(val_dataset)
    preds = np.argmax(raw_pred, axis=1)

    # Aggregate predictions per sample_id
    df = pd.DataFrame({
        'sample_id': val_sample_ids,
        'preds': preds,
        'labels': val_labels
    })
    aggregated = df.groupby('sample_id').agg({
        'preds': lambda x: np.argmax(np.bincount(x)),
        'labels': 'first'
    }).reset_index()

    acc = accuracy_score(aggregated['labels'], aggregated['preds'])
    print("Aggregated Accuracy:", acc)


if __name__ == "__main__":
    main()
