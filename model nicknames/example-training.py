import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Function to preprocess names and nicknames
def preprocess_names(data):
    # Combine First, Middle, and Surname into a single 'Name' field
    data['Name'] = data[['First', 'Middle', 'Surname']].apply(lambda x: ' '.join(x.dropna()), axis=1)
    # Combine Nicknames into a single string
    data['Nicknames'] = data['Nicknames'].apply(lambda x: ' '.join(x) if pd.notna(x) else '')
    return data[['Name', 'Nicknames']]



# Manually specify the encoding based on the result of the 'file' command
file_encoding = 'latin1'  # Replace with the correct encoding if known

# Use the specified encoding to read the CSV file
df = pd.read_csv("/Users/anil/AI/model nicknames/example-training.csv", encoding=file_encoding)



# Preprocess data
df_processed = preprocess_names(df)

# Assume you have a column named 'Label' for the actual labels in your dataset
y = df_processed['Label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert names to feature vectors using CountVectorizer
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train + ' ' + df_processed.loc[y_train, 'Nicknames'])
X_test_vec = vectorizer.transform(X_test + ' ' + df_processed.loc[y_test, 'Nicknames'])

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
