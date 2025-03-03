import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st  # Web-based UI

# Load trained LSTM model
model = tf.keras.models.load_model("single_layer_lstm_model.h5", compile=False)

# Manually compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Model loaded and compiled successfully!")

# Load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Sentiment Mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to preprocess input text
def preprocess_text(text, maxlen=100):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=maxlen, padding='post')
    return padded_seq

# Streamlit Web App
st.title("Twitter Sentiment Analysis App 🐦")

st.write("Enter a tweet below and analyze its sentiment using the trained LSTM model.")

# User Input
user_input = st.text_area("Enter a tweet...", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():  # Ensure input is not empty
        processed_text = preprocess_text(user_input)
        prediction = model.predict(processed_text)
        sentiment_label = np.argmax(prediction)
        st.write(f"**Predicted Sentiment:** {sentiment_mapping[sentiment_label]}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")



