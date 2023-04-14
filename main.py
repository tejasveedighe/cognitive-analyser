import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
import re
import pandas as pd
import csv

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the Sentiment140 dataset into a Pandas DataFrame
df = pd.read_csv('./dataset/Sentiment140.csv', header=None, names=[
                 'target', 'id', 'date', 'flag', 'user', 'text'], encoding='ISO-8859-1')


# Define a function for preprocessing the text data
def preprocess_text(text):
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text


# Apply the preprocessing function to the 'text' column of the DataFrame
print("Pre-process text called")
df['text'] = df['text'].apply(preprocess_text)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42)


# Vectorize the text data using a bag-of-words model
# print("Vectorizing the text data using bag-of-words")
# vectorizer = CountVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

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
print('Accuracy:', accuracy)

# Define a function to map the sentiment score to a depression score


def map_sentiment_to_depression(sentiment):
    if sentiment <= 0.33:
        return 'Depressed'
    elif sentiment <= 0.67:
        return 'Anxious'
    else:
        return 'Happy'

# Define a function to apply sentiment analysis to the text data and return the polarity score
def get_polarity_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Apply the sentiment analysis function to the 'text' column of the DataFrame
df['polarity'] = df['text'].apply(get_polarity_score)

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
print('Accuracy:', accuracy)

print("Hi there! I'm here to ask you some questions about your mental health. Please answer elaboratly to get more accurate analysis, explain as much as possible.")
profession = input(
    "Can you tell me your profession? (student, businessman, worker): ")

"""
Asks different questions based on the user's profession.
"""
questions = []
if profession == 'student':
    questions = ['How do you balance your studies and personal life?',
                 'Do you feel overwhelmed with schoolwork or exams?',
                 'Do you have a good support system to help you manage stress?',
                 "Do you feel comfortable talking about your mental health with someone at school (e.g., counselor, teacher, coach)?",
                 "Have you had trouble sleeping recently?",
                 "How often do you engage in physical activity or exercise?",
                 "Do you feel like you have a good support system of friends and family?",
                 "Have you experienced any feelings of hopelessness or worthlessness in the past month?",
                 "Do you have any concerns about your mental health that you would like to discuss with a professional?",
                 "How much stress do you feel related to your academic workload?",
                #  "Have you experienced any negative emotions related to school or academic performance, such as frustration or low self-esteem?",
                #  "How many hours per day do you typically spend on schoolwork or studying?",
                #  "Do you feel like you have a healthy balance between schoolwork and other activities?",
                #  "Have you been participating in any extracurricular activities or hobbies that you enjoy?",
                #  "Have you noticed any changes in your motivation or interest in school or activities?",
                #  "Have you been feeling socially isolated or disconnected from others at school or in your community?",
                #  "Have you experienced any conflicts or negative experiences with peers or authority figures at school?",
                #  "Do you feel like you have access to adequate resources and support to succeed academically?",
                #  "Do you feel like your mental health is being adequately supported by your school or academic environment?"
                ]

elif profession == 'businessman':
    questions = ['Do you feel stressed with your work responsibilities?',
                 'Do you take breaks during your workday?',
                 'How do you manage your work-life balance?',
                 "How do you manage stress in your day-to-day life as a businessman?",
                 "What are some warning signs of burnout or depression that you have experienced in the past, and how did you address them?",
                 "Have you sought out any mental health resources or support in the past? If so, what was your experience like?",
                 "What strategies do you use to maintain a healthy work-life balance?",
                 "Have you ever experienced anxiety or panic attacks related to work? If so, how did you cope with them?",
                 "How do you prioritize self-care and relaxation in your busy schedule?",
                 "Have you ever experienced feelings of inadequacy or imposter syndrome as a businessman? If so, how did you overcome them?",
                 "Have you ever encountered ethical dilemmas or conflicts in your business that have affected your mental health? If so, how did you manage them?",
                 "What steps do you take to manage difficult conversations or conflicts with colleagues or clients in a healthy and constructive way?",
                 "Do you have any advice for other businessmen who may be struggling with their mental health?"
                 ]

elif profession == 'worker':
    questions = ['Do you have a good work-life balance?',
                 'Do you feel supported by your colleagues and supervisors?',
                 'Do you feel stressed with your workload?',
                 "How do you prioritize self-care and stress management in your profession?",
                 "Have you ever experienced burnout or compassion fatigue in your work? If so, how did you address it?",
                 "What strategies do you use to maintain a healthy work-life balance?",
                 "Have you ever experienced discrimination or harassment in your workplace, and how did it affect your mental health?",
                 "What resources or support have you sought out for your mental health, and how have they helped?",
                 "How do you manage anxiety or other mental health challenges while performing your job duties?",
                 "Have you ever encountered ethical dilemmas or conflicts in your profession that have affected your mental health?, If so, how did you manage them?",
                 "How do you handle high-pressure situations or intense deadlines without negatively impacting your mental health?",
                 "Have you ever struggled with imposter syndrome or feelings of inadequacy in your profession? If so, how did you overcome them?",
                 "Do you have any advice for others in your profession who may be struggling with their mental health?,"
                 ]

# questions.extend(["How would you rate your overall mood in the past week?",
#                   "Have you felt any loss of interest or pleasure in things you normally enjoy?",
#                   "How often have you felt sad, down or hopeless in the past week?",
#                   "Have you had any trouble sleeping or sleeping too much?",
#                   "Have you experienced a loss of appetite or have you been overeating?",
#                   "Have you been more irritable or easily agitated than usual?",
#                   "On average, how many hours of sleep do you get each night?",
#                   "Have you experienced any physical symptoms, such as headaches or stomach aches, without a clear physical cause?",
#                   "Have you had any thoughts of self-harm or suicide?",
#                   "How would you rate your energy level in the past week?",
#                   "How would you rate your ability to concentrate in the past week?",
#                   "Have you felt anxious or nervous in the past week?",
#                   "On average, how many hours per day do you spend on social media?",
#                   "Have you experienced any recent stressful life events?",
#                   "Do you feel like you have a support system to help you cope with stress?",
#                   "Have you experienced any traumatic events in the past?",
#                   "Have you ever been diagnosed with a mental health condition?",
#                   "Are you currently receiving treatment for a mental health condition?",
#                   "Have you experienced any significant life changes recently?"])

# Collect the user's responses to mental health questions
responses = []

for question in questions:
    response = input(str(questions.index(question) + 1) + ". " + question)
    responses.append(response)

# Preprocess the user's responses
preprocessed_responses = [preprocess_text(response) for response in responses]

# Vectorize the preprocessed responses using the same vectorizer as before
vectorized_responses = vectorizer.transform(preprocessed_responses)

# Predict the depression score using the trained model
depression_score = model.predict(vectorized_responses)

print('Depression score:', depression_score)

import statistics
final_prediction = depression_score.tolist()

with open(f"final_ouput.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Questions", "Response", "Prediction"])
            for response in responses:
                writer.writerow([questions[responses.index(response)], response, final_prediction[responses.index(response)]])

final_prediction = statistics.mode(depression_score)
print("Final Prediction: ", final_prediction)