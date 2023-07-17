# import necessary packages
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Preprocessing and Feature Extraction

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Tokenization and lowercasing
        tokens = word_tokenize(text.lower())

        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    return ""  # Returning empty string for non-string values


# Sentiment Analysis
def train_sentiment_analysis_model(X, y):
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    model = SVC(kernel='linear')
    model.fit(X_vectorized, y)
    return model, vectorizer


# Sentiment Extraction
def extract_sentiment_word_or_phrase(text, sentiment, model, vectorizer):
    vectorized_text = vectorizer.transform([text])
    predicted_sentiment = model.predict(vectorized_text)[0]

    if predicted_sentiment == sentiment:
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Extract sentiment words or phrases
        sentiment_words = [tokens[i] for i in range(len(tokens)) if tokens[i] in vectorizer.get_feature_names_out()]

        if sentiment_words:
            return ' '.join(sentiment_words)

    return None


# Load train and test CSV data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_text_column = train_data['text']
train_sentiment_column = train_data['sentiment']
train_text_id_column = train_data['textID']

test_text_column = test_data['text']
test_sentiment_column = test_data['sentiment']
test_text_id_column = test_data['textID']

# Remove rows with missing values from train data
train_data = train_data.dropna(subset=['text', 'sentiment'])
train_text_column = train_data['text']
train_sentiment_column = train_data['sentiment']

# Preprocess train text column
preprocessed_train_text = [preprocess_text(str(text)) for text in train_text_column]

# Prepare training data for sentiment analysis model
X_train = preprocessed_train_text
y_train = train_sentiment_column

# Remove rows with missing values in X_train and y_train
X_train, y_train = zip(*((text, sentiment) for text, sentiment in zip(X_train, y_train) if text))

# Convert y_train to a NumPy array
y_train = np.array(y_train)

# Train sentiment analysis model
sentiment_model, vectorizer = train_sentiment_analysis_model(X_train, y_train)

# Test the model on test data and save the results to CSV
results = []
correct_predictions = 0
total_predictions = 0
for text_id, text, sentiment in zip(test_text_id_column, test_text_column, test_sentiment_column):
    sentiment_word_or_phrase = extract_sentiment_word_or_phrase(str(text), sentiment, sentiment_model, vectorizer)
    results.append({'textID': text_id, 'selected_text': sentiment_word_or_phrase})

results_df = pd.DataFrame(results)
results_df.to_csv('submission.csv', index=False)
