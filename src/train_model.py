import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Load the dataset
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# Display the first 5 rows
print(data.head())


# Input data
X = data["message"]


# Output labels
y = data["label"]


print("\nInput data:")
print(X.head())


print("\nOutput labels:")
print(y.head())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# Convert text messages into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# Create the machine learning model
model = LogisticRegression()


# Train the model
model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# Make predictions on the test data
predictions = model.predict(X_test_tfidf)

print("\nFirst 10 predictions:")
print(predictions[:10])


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", accuracy)
print("Model accuracy percentage:", accuracy * 100, "%")
# Detailed model evaluation
print("\nClassification Report:")
print(classification_report(y_test, predictions))
# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)
# Save the trained model
joblib.dump(model, "model/spam_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
