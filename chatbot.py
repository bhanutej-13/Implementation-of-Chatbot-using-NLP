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
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            st.error("Error loading intents. Please check the file and format.")
            st.stop()

    def load_or_train_model(self):
        """
        Loads the pre-trained model and vectorizer if available; otherwise, trains the model.
        """
        intents_file_mtime = os.path.getmtime("intents.json")
        model_exists = os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path)

        if model_exists:
            model_mtime = os.path.getmtime(self.model_path)
            vectorizer_mtime = os.path.getmtime(self.vectorizer_path)

            if intents_file_mtime < model_mtime and intents_file_mtime < vectorizer_mtime:
                self.clf = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                return

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
    log_path = "chat_log.csv"
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])
        csv_writer.writerow([user_input, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def about_section():
    """
    Displays the About section with information about the chatbot and the project.
    """
    st.title("About the Chatbot")
    st.markdown("""
    This chatbot is built using Natural Language Processing (NLP) and Machine Learning technologies:

    - **Streamlit**: For the web-based interface.
    - **Scikit-learn**: For text vectorization and intent classification.
    - **Joblib**: For model persistence.

    ### How It Works:
    1. User inputs a message.
    2. Logistic Regression model predicts the intent.
    3. Chatbot responds with a matching message from the intent.

    ### Future Enhancements:
    - Multi-language support
    - Integration with real-time APIs
    - Context-aware conversation handling
    """)

def display_chat_history():
    """
    Displays saved conversation history.
    """
    st.title("Conversation History")
    log_path = "chat_log.csv"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                st.markdown(f"**User:** {row[0]}  ")
                st.markdown(f"**Chatbot:** {row[1]}  ")
                st.caption(f"🕒 {row[2]}")
                st.divider()

        if st.button("Clear History"):
            os.remove(log_path)
            st.success("Conversation history cleared!")
    else:
        st.write("No conversation history available.")

def main():
    """
    Main function to run the chatbot interface.
    """
    st.set_page_config(page_title="NLP Chatbot", page_icon="🤖")

    st.sidebar.title("Menu")
    menu_options = ["Home", "About", "Conversation History"]
    choice = st.sidebar.selectbox("Choose an option", menu_options)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if choice == "Home":
        st.title("Chatbot using NLP")
        st.write("Start chatting below!")

        chatbot = Chatbot("intents.json")

        user_input = st.text_input("You:", key="user_input")
        if user_input:
            with st.spinner("Thinking..."):
                time.sleep(1)
                response = chatbot.get_response(user_input)

            st.session_state.conversation.append((user_input, response))
            save_conversation(user_input, response)

            for user_msg, bot_resp in st.session_state.conversation[-10:]:
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**Chatbot:** {bot_resp}")

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day!")
                st.session_state.conversation = []

    elif choice == "About":
        about_section()

    elif choice == "Conversation History":
        display_chat_history()

if __name__ == '__main__':
    main()
