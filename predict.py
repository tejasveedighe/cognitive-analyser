import statistics
from chatbot_gui import ChatbotGUI
import pandas as pd
import pickle
from pre_process import preprocess_text
import csv

# import the pre-trained model and vectorizer instance
model = pickle.load(open('svm_model.sav', 'rb'))
vectorizer = pickle.load(open('svm_vectorizer.pkl', 'rb'))

# calling the chatbot
bot = ChatbotGUI()

# Collect the user's responses to mental health questions exported by chatbot
responses = pd.read_csv('user_responses.csv')

responses = responses.values.tolist()

# Preprocess the user's responses
# skipping the first two values as they are header and name of user
preprocessed_responses = [preprocess_text(res[0]) for res in responses[2:]]
# print(preprocessed_responses)

# Vectorize the preprocessed responses using the same vectorizer as before
vectorized_responses = vectorizer.transform(preprocessed_responses)

# Predict the depression score using the trained model
depression_score = model.predict(vectorized_responses)
print('Depression score:', depression_score)

final_prediction = depression_score.tolist()
final_prediction = statistics.mode(final_prediction)

print("Final Prediction: ", final_prediction)
