import joblib

# Load the trained model
model = joblib.load("model/spam_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Get message from the user
message = input("Enter a message: ")

# Convert the message into TF-IDF features
message_tfidf = vectorizer.transform([message])

# Make prediction
prediction = model.predict(message_tfidf)

# Display result
if prediction[0] == "spam":
    print("\n🚨 SPAM MESSAGE")
else:
    print("\n✅ NOT SPAM")