import os
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
tqdm.pandas(desc='Preprocessing')
import nltk

# Télécharger les ressources nécessaires pour nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Prétraitement du texte
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'^rt\s+', '', text)  # Remove 'rt' at the beginning
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()  # Tokenization
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

# Charger les données et appliquer le prétraitement
def load_and_preprocess_data(folder_path):
    data = []
    print("Loading and preprocessing data...")
    for filename in tqdm(os.listdir(folder_path), desc="Files processed"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df = df.sample(n=10000, random_state=42) # On garde que 10000 tweets par match 
        df['Tweet'] = df['Tweet'].progress_apply(preprocess_text)
        data.append(df)
    return pd.concat(data, ignore_index=True)

# Classe pour encapsuler les données dans un format compatible PyTorch
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Prédiction sur un ensemble de données
def predict_on_eval_data(model, tokenizer, folder_path, output_path):
    eval_data = []
    print("Processing evaluation data...")
    for filename in tqdm(os.listdir(folder_path), desc="Files processed"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df['Tweet'] = df['Tweet'].progress_apply(preprocess_text)
        eval_data.append(df)

    eval_df = pd.concat(eval_data, ignore_index=True)
    eval_texts = eval_df['Tweet'].tolist()

    print("Tokenizing evaluation data...")
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    print("Making predictions...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_texts), 16), desc="Batches processed"):
            inputs = {key: val[i:i+16].to('cuda') for key, val in eval_encodings.items()}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().tolist()
            predictions.extend(preds)

    eval_df['Prediction'] = predictions
    eval_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    # Charger les données d'entraînement
    train_folder = "train_tweets"
    df = load_and_preprocess_data(train_folder)
    df = df.sample(n=5000, random_state=42)

    # Préparer les données
    print("Splitting data into training and validation sets...")
    tweets = df['Tweet'].tolist()
    labels = df['EventType'].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)

    # Charger le tokenizer et le modèle BERT
    print("Loading tokenizer and BERT model...")
    bert_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)


    # Tokenisation
    print("Tokenizing training and validation data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Créer les datasets
    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    # Configurer les paramètres d'entraînement
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Évaluation à chaque époque
        save_strategy="epoch",        # Sauvegarde à chaque époque pour correspondre
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        fp16=True,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )


    # Définir le Trainer
    print("setting un trainer")
    trainer = Trainer(
        model=model,                          
        args=training_args,                   
        train_dataset=train_dataset,          
        eval_dataset=val_dataset,             
        tokenizer=tokenizer,                  
    )

    # Entraîner le modèle
    print("Starting training...")
    trainer.train()

    # Sauvegarder le modèle
    print("Saving the fine-tuned model...")
    model.save_pretrained("./fine_tuned_bert")
    tokenizer.save_pretrained("./fine_tuned_bert")

    # Évaluer le modèle
    print("Evaluating the model...")
    metrics = trainer.evaluate()
    print("Metrics:", metrics)

    # Prédictions sur le dossier d'évaluation
    # eval_folder = "eval_tweets"
    # output_file = "eval_predictions.csv"
    # predict_on_eval_data(model, tokenizer, eval_folder, output_file)

if __name__ == "__main__":
    main()
