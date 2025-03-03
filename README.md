# Twitter Sentiment project
 A deep learning-based sentiment analysis model that classifies tweets as Positive, Neutral, or Negative using LSTM (Long Short-Term Memory) networks.
 This project aims to analyze sentiments in tweets using an LSTM-based neural network. The model is trained on a preprocessed Twitter dataset and optimized to handle imbalanced classes using techniques like class weighting and SMOTE oversampling.
📂 Project Structure:-
📁 Twitter-Sentiment-Analysis  
│── 📜 twitter_sentiment_app.py  # Streamlit-based web app  
│── 📜 train_lstm_model.py       # LSTM training script  
│── 📜 preprocess_data.py        # Data cleaning and preprocessing  
│── 📜 sentiment_model.h5        # Trained LSTM model  
│── 📜 tokenizer.pickle          # Tokenizer for text processing  
│── 📜 accuracy_results.csv      # Training & validation accuracy logs  
│── 📊 glove.6B.100d.txt         # Pretrained GloVe embeddings  
│── 📄 README.md                 # Project documentation  
