# Sentiment Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohitmethi1000/sentiment-analysis/blob/main/sentiment_analysis.ipynb)


This project performs sentiment analysis on Amazon product reviews using both a rule-based approach (VADER) and a transformer-based deep learning model (RoBERTa). The goal is to analyze the emotional tone of customer reviews and compare different sentiment analysis techniques.

## Features

- Load and preprocess Amazon review data (text and star ratings)
- Perform sentiment analysis using:
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - getting a roBERTa-base model via HuggingFace Transformers
- Tokenize and extract POS and named entities using NLTK
- Apply softmax-normalized sentiment scoring using Roberta
- Visualize sentiment scores across different models and star ratings
- Identify polarity-star mismatches to detect noisy labels or sarcasm

## Models Used

- **VADER**: Rule-based sentiment model using a predefined lexicon
- **RoBERTa**: Context-aware transformer model fine-tuned on Twitter data, accessed via HuggingFace

## Visualization

- Bar plots showing compound/positive/neutral/negative scores across star ratings
- Pair plots comparing VADER and Roberta sentiment outputs
- Examples of mismatched reviews: positive tone with 1-star or negative tone with 5-stars

## Tech Stack

- Python, Pandas, NumPy
- NLTK for NLP preprocessing
- HuggingFace Transformers for Roberta
- Seaborn and Matplotlib for visualization
- tqdm for progress bars

