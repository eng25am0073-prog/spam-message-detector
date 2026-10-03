import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model/spam_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Page title
st.title("📱 Spam Message Detector")

st.write(
    "Enter an SMS message below and the machine learning model "
    "will predict whether it is spam or not."
)

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize!"
)

# Prediction button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Convert message into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_tfidf)

        if prediction[0] == "spam":
            st.error("🚨 SPAM MESSAGE")

        else:
            st.success("✅ NOT SPAM")