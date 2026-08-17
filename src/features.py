import re
import pandas as pd

# Standard Libraries
import os
import re
import logging
import warnings
from collections import Counter
from typing import Dict
from email.parser import Parser
from pathlib import Path

# Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Deep Learning
import torch
from transformers import pipeline
from tqdm import tqdm

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# Your important columns
subject = "subject_clean"
target = "transformer_label"

pos_class = "spam"
neg_class = "ham"
labels_ = [neg_class, pos_class]

# Important numeric features
numeric_cols = [
    "subject_char_len_after",
    "subject_word_count_after",
]


def make_text_features():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
        lowercase=True,
    )

    return FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])


def make_preprocess():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    return ColumnTransformer([
        ("text", make_text_features(), "text"),
        ("num", numeric_pipe, numeric_cols),
    ])



def replace_numbers(match):
    num = match.group()

    # percentage
    if '%' in num:
        return ' percent '

    # money
    if '$' in num:
        return ' money '

    # extract digits only
    digits = re.sub(r'\D', '', num)
    if not digits:
        return ' number '

    value = int(digits)

    if value < 10:
        return ' smallnum '
    elif value < 100:
        return ' mediumnum '
    elif value < 1000:
        return ' bignum '
    else:
        return ' hugenum '


def clean_subject(text: str) -> str:
    """
    Clean and preprocess email subject text.
    """
    # Handle missing values
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''

    text = text.strip()

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # replace numbers
    text = re.sub(r'\$?\d[\d,]*(\.\d+)?%?', replace_numbers, text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens
              if token not in STOP_WORDS and len(token) > 1]

    # Join back to string
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def build_single_input(subject_text):
    cleaned = clean_subject(subject_text)

    row = {
        "text": cleaned,
        # "subject_char_len_before": len(str(subject_text)),
        # "subject_word_count_before": len(str(subject_text).split()),
        "subject_char_len_after": len(cleaned),
        "subject_word_count_after": len(cleaned.split()),
        # "has_cc": 0,
        # "has_bcc": 0,
        # "to_count": 1,
        # "cc_count": 0,
        # "bcc_count": 0,
        # "content_type": "unknown",
        # "content_transfer_encoding": "unknown",
    }

    return pd.DataFrame([row])
