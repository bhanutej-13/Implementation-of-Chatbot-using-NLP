import os
import json
import random
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import csv
from datetime import datetime

class Chatbot:
    def __init__(self, intents_file):
        """
        Initializes the chatbot by loading intents and training the model.
        """
        self.intents = self.load_intents(intents_file)
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(random_state=0, max_iter=10000)
        self.model_path = "chatbot_model.joblib"
        self.vectorizer_path = "vectorizer.joblib"
        self.load_or_train_model()

    def load_intents(self, file_path):
        """
        Loads intents from a JSON file.
        """
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            st.error("The intents file was not found. Please ensure 'intents.json' exists.")
            st.stop()
        except json.JSONDecodeError:
            st.error("The intents file is not valid JSON. Please check the file format.")
            st.stop()

    def load_or_train_model(self):
        """
        Loads the pre-trained model and vectorizer if they exist; otherwise, trains the model.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.clf = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            self.train_model()

    def train_model(self):
        """
        Trains the chatbot model using the intents data.
        """
        tags, patterns = [], []
        for intent in self.intents:
            for pattern in intent['patterns']:
                tags.append(intent['tag'])
                patterns.append(pattern)
        
        x = self.vectorizer.fit_transform(patterns)
        y = tags
        self.clf.fit(x, y)
        
        # Save the trained model and vectorizer
        joblib.dump(self.clf, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

    def get_response(self, input_text):
        """
        Generates a response for the given user input.
        """
        input_text = self.vectorizer.transform([input_text])
        tag = self.clf.predict(input_text)[0]
        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return "I'm sorry, I don't understand that. Can you rephrase?"

def save_conversation(user_input, response):
    """
    Saves the conversation to a CSV file for history.
    """
    with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([user_input, response, timestamp])

def about_section():
    """
    Displays the About section with information about the chatbot and the project.
    """
    st.title("About the Chatbot")
    st.write("""
    This chatbot is built using Natural Language Processing (NLP) techniques to understand and respond to user inputs based on predefined intents. 
    It uses the following technologies:
    - **Streamlit**: For the web-based interface.
    - **Scikit-learn**: For text vectorization and intent classification.
    - **Joblib**: For saving and loading the trained model.
    """)

    st.subheader("How It Works")
    st.write("""
    1. The chatbot is trained on a dataset of intents, patterns, and responses stored in `intents.json`.
    2. When a user inputs a message, the chatbot uses a trained Logistic Regression model to predict the intent.
    3. Based on the predicted intent, the chatbot selects a random response from the corresponding intent's responses.
    """)

    st.subheader("Project Goals")
    st.write("""
    - To create a simple yet effective chatbot that can handle basic user queries.
    - To demonstrate the use of NLP and machine learning in building conversational agents.
    - To provide a user-friendly interface for interacting with the chatbot.
    """)

    st.subheader("Future Enhancements")
    st.write("""
    - Add support for multi-language conversations.
    - Integrate with external APIs for real-time information (e.g., weather, news).
    - Improve the chatbot's ability to handle complex and context-aware conversations.
    """)

def main():
    """
    Main function to run the Streamlit chatbot interface.
    """
    st.sidebar.title("Menu")
    menu_options = ["Home", "About", "Conversation History"]
    choice = st.sidebar.selectbox("Choose an option", menu_options)

    if choice == "Home":
        st.title("Chatbot using NLP")
        st.write("Welcome to the chatbot. Type a message and press Enter to start the conversation.")

        # Initialize chatbot
        chatbot = Chatbot("intents.json")

        # User input and chatbot response
        user_input = st.text_input("You:", key="user_input")
        if user_input:
            with st.spinner("Thinking..."):
                time.sleep(1)  # Simulate processing time
                response = chatbot.get_response(user_input)
            
            st.text_area("Chatbot:", value=response, height=120)
            save_conversation(user_input, response)

            # Stop the chatbot if the user says goodbye
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "About":
        about_section()

    elif choice == "Conversation History":
        st.title("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

if __name__ == '__main__':
    main()  