# Project: GloVe-based Neural Network for Tweet Analysis

This repository contains a set of tools and models for analyzing tweets using pre-trained GloVe embeddings. It provides utilities for preprocessing tweet datasets, generating embeddings, training machine learning models, and evaluating their performance.

This repository was built for a kaggle challenge for the CSC_51054_EP - Machine and Deep learning (2024-2025) course at Polytechnique. The challenge was about binary classification problem: our
mission was to build a model that can accurately predict the occurrence of notable events
within specified one-minute intervals of a football match.

## Repository Structure

### Files

1. **`tools.py`**
   - **Purpose**: Contains utility functions for preprocessing tweets and creating embeddings.
   - **Key Functions**:
     - `preprocess_tweet(tweet)`: Cleans and normalizes tweets by removing special characters, URLs, mentions, and stopwords. Converts emojis to descriptive text and lemmatizes words.
     - `get_avg_embedding(tweet, model, vector_size=200)`: Computes the average word vector for a tweet using a pre-trained word embedding model.
     - `create_subdivisions_with_concatenation(df, num_subdivisions, embeddings_model)`: Divides tweet data into subdivisions, computes average embeddings, and concatenates them into a single feature vector.
     - `prepare_eval_data_with_concatenation(eval_df, num_subdivisions, embeddings_model)`: Processes evaluation data by creating subdivisions and computing embeddings.

2. **`glove_nn.py`**
   - **Purpose**: Implements the neural network model and training loop for tweet analysis.
   - **Key Components**:
     - `NeuralNet`: A multi-layer fully connected neural network with batch normalization, dropout, and Leaky ReLU activation functions.
     - `train(num_epochs, model, train_loader, criterion, optimizer, scheduler=None)`: Training loop for optimizing the neural network model.

3. **`glove_preprocess.py`**
   - **Purpose**: Loads, preprocesses, and splits tweet datasets for training, testing, and evaluation.
   - **Key Features**:
     - Loads GloVe embeddings using Gensim's API (`glove-twitter-200`).
     - Preprocesses tweet datasets using `tools.py` utilities.
     - Computes and saves sentiment scores using TextBlob.
     - Saves processed datasets as `.pkl` files for further use.

4. **`ML_tests.ipynb`**
   - **Purpose**: Contains experiments with multiple machine learning models for analyzing tweets.
   - **Key Features**:
     - Loads preprocessed data from the `.pkl` files generated by `glove_preprocess.py`.
     - Implements various ML models, including the `NeuralNet` from `glove_nn.py`.
     - Compares the performance of different models using metrics and visualizations.

## Workflow

### 1. Preprocessing Tweets
Use the functions in `tools.py` and `glove_preprocess.py` to clean and preprocess tweets. These steps include:
- Normalizing text (e.g., removing URLs, special characters).
- Computing GloVe embeddings for each tweet.
- Splitting data into training, testing, and evaluation sets.

### 2. Training the Neural Network
- Use `glove_nn.py` to define and train the `NeuralNet` model.
- The `train` function handles training with options for learning rate scheduling.

### 3. Experimenting with Models
- Use `ML_tests.ipynb` to test and compare various machine learning models, including:
  - Neural networks from `glove_nn.py`.
  - Other ML algorithms applied to the preprocessed data.

### 4. Evaluating Performance
- Evaluate models using standard metrics such as accuracy, precision, recall, and F1-score.
- Use the evaluation datasets processed by `glove_preprocess.py`.

## Prerequisites

### Python Packages
- `numpy`
- `pandas`
- `torch`
- `gensim`
- `nltk`
- `textblob`
- `emoji`
- `tqdm`

### Data
- The project requires GloVe embeddings, which are downloaded using Gensim's API.
- Tweet datasets should be formatted as CSV files with relevant columns (`Tweet`, `MatchID`, `PeriodID`, etc.).

### Hardware
- A GPU is recommended for training the neural network.

## Running the Project

1. Preprocess datasets:
   ```bash
   python glove_preprocess.py
   ```
   This will save preprocessed data and embeddings as `.pkl` files. Since the csv files containing all the tweets are too heavy, only samples are provided. But the pkl files available were generated using all the available data, so you can run the notebook without any problem.

2. Train the neural network:
   ```bash
   python glove_nn.py
   ```

3. Test models and visualize results:
   - Open `ML_tests.ipynb` in Jupyter Notebook.
   - Run the cells to compare model performance.

## Outputs
- Preprocessed datasets: `train_subdivided_data.pkl`, `test_subdivided_data.pkl`, `eval_subdivided_data.pkl`
- Trained models: Neural network weights (if saved).
- Performance metrics and visualizations: Available in `ML_tests.ipynb`.

## Future Improvements
- Implement additional NLP features such as part-of-speech tagging.
- Experiment with advanced neural architectures (e.g., transformers).
