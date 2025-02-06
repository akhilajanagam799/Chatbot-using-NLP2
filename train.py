import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download necessary NLP resources
nltk.download("punkt")
nltk.download("wordnet")

# Load the chatbot dataset
with open("intents.json", "r") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare training data
X_train = []
y_train = []
labels = []
responses = {}

for idx, intent in enumerate(data["intents"]):
    labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        X_train.append(" ".join(tokens))
        y_train.append(idx)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
y_train = np.array(y_train)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(labels, open("labels.pkl", "wb"))
pickle.dump(responses, open("responses.pkl", "wb"))

print(" Chatbot training completed! Model saved.")