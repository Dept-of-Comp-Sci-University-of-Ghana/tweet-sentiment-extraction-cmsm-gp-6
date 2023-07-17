import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess the data (e.g., removing quotes, converting labels)

# Handle missing values
train_df.fillna('', inplace=True)


# Split the training data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['selected_text'], test_size=0.2, random_state=42)

# Create feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.astype(str))  # Convert to string to handle missing values
X_val_vec = vectorizer.transform(X_val.astype(str))  # Convert to string to handle missing values

# Train a classifier model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val_vec)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Use the trained model to make predictions on the test set
X_test_vec = vectorizer.transform(test_df['text'].astype(str))  # Convert to string to handle missing values
test_predictions = model.predict(X_test_vec)

#  Format the predictions and create the submission file
submission_df = pd.DataFrame({'textID': test_df['textID'], 'selected_text': test_predictions})
submission_df.to_csv('submission.csv', index=False)
