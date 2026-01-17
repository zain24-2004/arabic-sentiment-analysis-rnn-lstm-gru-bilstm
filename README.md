# arabic-sentiment-analysis-rnn-lstm-gru-bilstm
Arabic Twitter sentiment analysis using deep learning sequence models (RNN, LSTM, GRU, and BiLSTM) trained on the Arabic Sentiment Twitter Corpus.
# Arabic Sentiment Analysis Using RNN, LSTM, GRU, and BiLSTM

This repository presents a deep learning–based approach for **Arabic sentiment analysis** using **Recurrent Neural Network (RNN)** architectures, including **LSTM, GRU, and Bidirectional LSTM (BiLSTM)**. The models are trained and evaluated on the **Arabic Sentiment Twitter Corpus**.

---

##  Project Overview

Sentiment analysis for Arabic text is challenging due to dialect diversity, morphology, and informal writing styles on social media.  
This project explores how different sequence-based deep learning models handle Arabic text data and compares their performance on sentiment classification.

---

##  Objectives

- Perform sentiment analysis on Arabic tweets
- Implement RNN, LSTM, GRU, and BiLSTM models
- Compare model performance on the same dataset
- Analyze the effect of model architecture on Arabic NLP tasks

---

##  Dataset

- **Name:** Arabic Sentiment Twitter Corpus  
- **Source:** Kaggle (`mksaad/arabic-sentiment-twitter-corpus`)
- **Content:** Arabic tweets labeled with sentiment
- **Classes:** Positive / Negative (and Neutral if available)

Dataset is downloaded using `kagglehub`.

---

##  Data Preprocessing

- Arabic text cleaning
- Removing URLs, symbols, and non-Arabic characters
- Tokenization
- Sequence padding
- Label encoding

---

##  Models Implemented

- Simple RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional LSTM (BiLSTM)

---

##  Methodology

1. Dataset Download
2. Text Cleaning & Normalization
3. Tokenization & Padding
4. Model Training
5. Model Evaluation
6. Architecture Comparison

---

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Loss & Accuracy Curves

---

##  Tools & Technologies

- Python
- TensorFlow / Keras
- Pandas & NumPy
- Matplotlib
- Scikit-learn
- KaggleHub
- Jupyter Notebook

---

## 📂 Project Structure

