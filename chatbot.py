import pickle
import nltk
import random
from nltk.stem import WordNetLemmatizer

# Load the trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
responses = pickle.load(open("responses.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def chatbot_response(user_input):
    processed_input = preprocess(user_input)
    X_test = vectorizer.transform([processed_input])
    
    try:
        tag = labels[model.predict(X_test)[0]]
        return random.choice(responses[tag])
    except:
        return "I'm sorry, I didn't understand that."

# Run chatbot in the terminal
print(" Chatbot is ready! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye! Have a great day! 👋")
        break
    print(f"Chatbot: {chatbot_response(user_input)}")