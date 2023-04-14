import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
import re
import pandas as pd
import pickle
from pre_process import preprocess_text


# Load the Sentiment140 dataset into a Pandas DataFrame
df = pd.read_csv('./dataset/Sentiment141.csv', header=None, names=[
                 'target', 'id', 'date', 'flag', 'user', 'text'], encoding='ISO-8859-1')

# Check if the dataset is balanced i.e almost similar number of tweets that are positive and negative
# df['target'].value_counts().plot(kind='pie')

# Apply the preprocessing function to the 'text' column of the DataFrame
print("Pre-processing the tweets")
df['text'] = df['text'].apply(preprocess_text)

print("Splitting the data into training and testing sets")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42)


print("Vectorizing the data using TF-IDF")
# Alternatively, vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a binary classification model on the vectorized text data
model = LinearSVC()
model.fit(X_train_vectorized, y_train)

# Predict the labels for the testing set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model without polarity and sentiment analysis of tweets:', accuracy)

# Define a function to map the sentiment score to a depression score
def map_sentiment_to_depression(sentiment):
    if sentiment < 0.2:
        return 'Depressed'
    elif sentiment < 0.4:
        return 'Anxious'
    elif sentiment < 0.6:
        return 'Neutral'
    elif sentiment < 0.8:
        return 'Happy'
    elif sentiment < 0.9:
        return 'Excited'
    else:
        return 'Very Happy'


# Define a function to apply sentiment analysis to the text data and return the polarity score
def get_polarity_score(text):
    blob = TextBlob(text)
    # print(blob.sentiment)
    return blob.sentiment.polarity

print("Getting the polarity of the text")
# Apply the sentiment analysis function to the 'text' column of the DataFrame
df['polarity'] = df['text'].apply(get_polarity_score)

print("Mapping the polarity of columns to the dataframe")
# Apply the function to the 'polarity' column of the DataFrame
df['depression'] = df['polarity'].apply(map_sentiment_to_depression)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['depression'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a multi-class classification model on the vectorized text data
model = LinearSVC()
model.fit(X_train_vectorized, y_train)

# Predict the labels for the testing set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model with polarity and sentiments mapped to tweets:', accuracy)

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
# print(confusion_matrix(y_test, y_pred))

print("Saving the model to local")
filename = 'svm_model.sav'
pickle.dump(model, open(filename, 'wb'))

print("Saving the vectorizer instance to local")
pickle.dump(vectorizer, open('svm_vectorizer.pkl', 'wb'))
