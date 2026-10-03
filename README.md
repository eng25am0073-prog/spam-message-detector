# 📱 Spam Message Detector

A Machine Learning project that classifies SMS messages as Spam or Not Spam.

## 📌 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning
to detect whether an SMS message is spam.

The system takes a message as input, converts the text into numerical
features using TF-IDF, and uses Logistic Regression to classify the message.

## 🚀 Features

- SMS spam classification
- TF-IDF text vectorization
- Logistic Regression machine learning model
- Accuracy, Precision, Recall and F1-score evaluation
- Confusion Matrix
- Command-line prediction
- Streamlit web interface

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Matplotlib
- Seaborn
- Git & GitHub

## 📊 Dataset

The project uses the SMS Spam Collection dataset.

The dataset contains SMS messages labeled as:

- `ham` — normal message
- `spam` — spam message

## 🧠 Machine Learning Workflow

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Spam / Not Spam Prediction