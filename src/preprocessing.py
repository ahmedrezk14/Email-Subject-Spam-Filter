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

# Configuration
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_nltk_data():
    # Download NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

    print("✓ All dependencies installed successfully!")


download_nltk_data()

# Initialize preprocessing tools
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def parse_email_message(raw_message: str) -> Dict:
    """
    Parse a raw email message and extract key fields.
    """
    email_dict = {
        'subject': '',
        # 'message_id': '',
        # 'date': '',
        # 'from_email': '',
        # 'to_email': '',
        # 'cc': '',
        # 'bcc': '',
        # 'mime_version': '',
        # 'content_type': '',
        # 'content_transfer_encoding': '',
        # 'x_from': '',
        # 'x_to': '',
        # 'x_cc': '',
        # 'x_bcc': '',
        # 'x_folder': '',
        # 'x_origin': '',
        # 'x_filename': '',
        # 'body': ''
    }

    try:
        # Split headers from body
        if '\n\n' in raw_message:
            headers_part, body_part = raw_message.split('\n\n', 1)
        else:
            headers_part = raw_message
            body_part = ''

        # Parse headers
        parser = Parser()
        msg = parser.parsestr(headers_part)

        # Extract fields
        email_dict['subject'] = msg.get('Subject', '')
        # email_dict['message_id'] = msg.get('Message-ID', '')
        # email_dict['date'] = msg.get('Date', '')
        # email_dict['from_email'] = msg.get('From', '')
        # email_dict['to_email'] = msg.get('To', '')
        # email_dict['cc'] = msg.get('Cc', '')
        # email_dict['bcc'] = msg.get('Bcc', '')
        # email_dict['mime_version'] = msg.get('Mime-Version', '')
        # email_dict['content_type'] = msg.get('Content-Type', '')
        # email_dict['content_transfer_encoding'] = msg.get('Content-Transfer-Encoding', '')
        # email_dict['x_from'] = msg.get('X-From', '')
        # email_dict['x_to'] = msg.get('X-To', '')
        # email_dict['x_cc'] = msg.get('X-cc', '')
        # email_dict['x_bcc'] = msg.get('X-bcc', '')
        # email_dict['x_folder'] = msg.get('X-Folder', '')
        # email_dict['x_origin'] = msg.get('X-Origin', '')
        # email_dict['x_filename'] = msg.get('X-FileName', '')
        # email_dict['body'] = body_part.strip()

    except Exception as e:
        logger.warning(f"Error parsing email: {str(e)[:100]}")
        email_dict['body'] = raw_message

    return email_dict


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


def get_top_ngrams(texts: list, n: int = 2, top_k: int = 20):
    all_ngrams = []
    for text in texts:
        words = str(text).split()
        text_ngrams = list(ngrams(words, n))
        all_ngrams.extend(text_ngrams)

    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_k)


def run_preprocessing():
    # =========================
    # Environment Check
    # =========================
    print(" All libraries imported successfully!")
    print(f" PyTorch GPU available: {torch.cuda.is_available()}")
    print(" Cleaning function defined")

    email_dataset_path = PROJECT_ROOT / 'data' / 'raw' / 'emails.csv'
    if not os.path.exists(email_dataset_path):
        import kagglehub
        email_dataset_path = kagglehub.dataset_download('wcukierski/enron-email-dataset')

        print(os.listdir(email_dataset_path))

        email_dataset_path = os.path.join(email_dataset_path, 'emails.csv')
        print(email_dataset_path)

        print('Data source import complete.')
    else:
        print('Dataset Path:', email_dataset_path)

    print("Loading dataset...")
    try:
        data = pd.read_csv(email_dataset_path)
        print(f"Dataset Loaded Successfully")
    except FileNotFoundError:
        print("Dataset not found. Download from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
        return

    # Display basic info
    print(f"\nDataset Shape: {data.shape}")
    print(f"\nColumn Names: {data.columns.tolist()}")
    print(f"\nMissing Values:")
    print(data.isnull().sum())
    print(f"\nDuplicate Rows: {data.duplicated().sum()}")

    # Display sample rows
    print("Sample Rows:")
    print("\n" + "="*80)
    for idx in range(min(2, len(data))):
        print(f"\nRow {idx}:")
        print(f"File: {data.iloc[idx]['file']}")
        print(f"Message (first 300 chars):\n{data.iloc[idx]['message'][:300]}...\n")
        print("="*80)

    print(f"Parsing {len(data)} emails...")

    parsed_emails = []
    for idx, row in data.iterrows():
        if idx % 10000 == 0:  # type: ignore
            print(f"Parsed {idx} / {len(data)} emails")

        parsed = parse_email_message(row["message"])
        parsed["file"] = row["file"]
        parsed_emails.append(parsed)

    # Create dataframe
    columns_order = [
        "subject",
        # "file",
        # "message_id",
        # "date",
        # "from_email",
        # "to_email",
        # "cc",
        # "bcc",
        # "mime_version",
        # "content_type",
        # "content_transfer_encoding",
        # "x_from",
        # "x_to",
        # "x_cc",
        # "x_bcc",
        # "x_folder",
        # "x_origin",
        # "x_filename",
        # "body",
    ]

    emails_df = pd.DataFrame(parsed_emails)
    emails_df = emails_df[columns_order]

    print(f"Parsing complete!")
    print(f"\nParsed Emails Shape: {emails_df.shape}")
    print(f"\nColumns: {emails_df.columns.tolist()}")

    # Display parsed emails info
    print(f"\nDataset Shape: {data.shape}")
    print(f"\nColumn Names: {data.columns.tolist()}")

    print(f"\nMissing Values per Column:")
    print(emails_df.isnull().sum())

    print(f"\nEmpty Subjects: {(emails_df['subject'] == '').sum() + emails_df['subject'].isnull().sum()}")
    print(f"Duplicated Subjects: {emails_df['subject'].duplicated().sum()}")

    print(f"\nSample Parsed Emails:")
    print(emails_df[['subject']].head(10))

    print(f"Initial shape: {emails_df.shape}")

    # Empty vs non-empty subjects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    empty_count = (emails_df['subject'].isna().sum() + (emails_df['subject'] == '').sum())
    non_empty_count = len(emails_df) - empty_count

    # Bar chart
    categories = ['Non-Empty', 'Empty']
    counts = [non_empty_count, empty_count]
    colors = ['lightblue', 'lightcoral']

    ax1.bar(categories, counts, color=colors)
    ax1.set_title('Subject Lines: Empty vs Non-Empty', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    for i, count in enumerate(counts):
        ax1.text(i, count, str(count), ha='center', va='bottom')
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Subject Lines Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"Non-empty subjects: {non_empty_count}")
    print(f"Empty subjects: {empty_count}")

    # Count rows before cleaning
    rows_before = len(emails_df)

    # Detect empty subjects
    empty_subject_mask = emails_df['subject'].str.strip() == ''

    # Keep only non-empty subjects
    emails_df = emails_df[~empty_subject_mask].copy()

    # Count removed rows
    removed_empty_subjects = rows_before - len(emails_df)

    print(f"Removed {removed_empty_subjects} empty subject rows")
    print(f"Current shape: {emails_df.shape}")

    # Count rows before duplicate removal
    rows_before = len(emails_df)

    # Remove duplicate subjects
    emails_df.drop_duplicates(subset=['subject'], inplace=True)

    # Count removed duplicates
    removed_duplicates = rows_before - len(emails_df)

    print(f"Removed {removed_duplicates} duplicate subjects")
    print(f"Current shape: {emails_df.shape}")

    emails_df.reset_index(drop=True, inplace=True)

    print("Index reset complete")
    print(f"Final shape: {emails_df.shape}")

    # Display sample subjects
    print(emails_df[['subject']].head())

    # Missing values visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    missing = emails_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        missing.plot(kind='bar', ax=ax, color='coral')
    else:
        print("No missing values to plot")
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Missing values visualization complete")

    # Subject length distribution
    subject_lengths = [len(str(s).split()) if isinstance(s, str) else 0 for s in emails_df['subject']]
    subject_chars = [len(str(s)) if isinstance(s, str) else 0 for s in emails_df['subject']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Word count
    ax1.hist(subject_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Subject Line Word Count Distribution (Before Cleaning)', fontsize=12, fontweight='bold')
    ax1.axvline(np.mean(subject_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(subject_lengths):.1f}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Character count
    ax2.hist(subject_chars, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Character Count')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Subject Line Character Count Distribution (Before Cleaning)', fontsize=12, fontweight='bold')
    ax2.axvline(np.mean(subject_chars), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(subject_chars):.1f}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Average subject length: {np.mean(subject_lengths):.1f} words, {np.mean(subject_chars):.1f} chars")

    # Create preprocessing columns
    print("Creating preprocessing columns...")

    # Statistics before cleaning
    emails_df['subject_char_len_before'] = emails_df['subject'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    emails_df['subject_word_count_before'] = emails_df['subject'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    # Clean
    print("Cleaning subjects...")
    emails_df['subject_clean'] = emails_df['subject'].apply(clean_subject)

    # Statistics after cleaning
    emails_df['subject_char_len_after'] = emails_df['subject_clean'].apply(lambda x: len(x))
    emails_df['subject_word_count_after'] = emails_df['subject_clean'].apply(lambda x: len(x.split()) if len(x) > 0 else 0)

    print(f"✓ Preprocessing complete")

    # Drop rows with empty cleaned subjects
    initial_count = len(emails_df)
    emails_df = emails_df[emails_df['subject_clean'].str.len() > 0].copy()
    removed_count = initial_count - len(emails_df)

    print(f"Removed {removed_count} rows with empty cleaned subjects")
    print(f"Final shape: {emails_df.shape}")

    emails_df = emails_df.reset_index(drop=True)

    # Remove duplicate cleaned subjects
    print("\nRemoving duplicate cleaned subjects...")
    rows_before_dedup = len(emails_df)
    emails_df.drop_duplicates(subset=['subject_clean'], inplace=True)
    removed_duplicates = rows_before_dedup - len(emails_df)

    print(f"Removed {removed_duplicates} duplicate cleaned subjects")
    print(f"Final shape after deduplication: {emails_df.shape}")

    emails_df = emails_df.reset_index(drop=True)

    # Display preprocessing statistics
    print("Preprocessing Statistics:")
    print(f"Total subjects: {len(emails_df)}")
    print(f"\nBefore Cleaning:")
    print(f"  Average characters: {emails_df['subject_char_len_before'].mean():.1f}")
    print(f"  Average words: {emails_df['subject_word_count_before'].mean():.1f}")
    print(f"\nAfter Cleaning:")
    print(f"  Average characters: {emails_df['subject_char_len_after'].mean():.1f}")
    print(f"  Average words: {emails_df['subject_word_count_after'].mean():.1f}")

    # Show examples
    print(f"\nExamples Before/After Cleaning:")
    for idx in range(min(5, len(emails_df))):
        print(f"\nBefore: {emails_df.iloc[idx]['subject']}")
        print(f"After:  {emails_df.iloc[idx]['subject_clean']}")

    # Text length distribution after cleaning
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Word count after cleaning
    ax1.hist(emails_df['subject_word_count_after'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Subject Line Word Count Distribution (After Cleaning)', fontsize=12, fontweight='bold')
    ax1.axvline(emails_df['subject_word_count_after'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {emails_df["subject_word_count_after"].mean():.1f}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Character count after cleaning
    ax2.hist(emails_df['subject_char_len_after'], bins=50, color='lightyellow', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Character Count')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Subject Line Character Count Distribution (After Cleaning)', fontsize=12, fontweight='bold')
    ax2.axvline(emails_df['subject_char_len_after'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {emails_df["subject_char_len_after"].mean():.1f}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Most frequent words after cleaning
    all_words = ' '.join(emails_df['subject_clean']).split()
    word_freq = Counter(all_words).most_common(30)

    words = [word for word, count in word_freq]
    counts = [count for word, count in word_freq]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(words)), counts, color='skyblue')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Frequency')
    ax.set_title('Top 30 Most Frequent Words in Cleaned Subjects', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Most frequent words: {words[:10]}")

    # Bigrams
    top_bigrams = get_top_ngrams(emails_df['subject_clean'], n=2, top_k=20)

    bigram_labels = [' '.join(bg[0]) for bg in top_bigrams]
    bigram_counts = [bg[1] for bg in top_bigrams]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(bigram_labels)), bigram_counts, color='lightgreen')
    ax.set_yticks(range(len(bigram_labels)))
    ax.set_yticklabels(bigram_labels)
    ax.set_xlabel('Frequency')
    ax.set_title('Top 20 Bigrams in Cleaned Subjects', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Top bigrams: {bigram_labels[:5]}")

    # Word cloud of cleaned subjects
    fig, ax = plt.subplots(figsize=(14, 8))

    all_subjects = ' '.join(emails_df['subject_clean'])
    wordcloud = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate(all_subjects)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Cleaned Email Subjects', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

    print("Word cloud created successfully")

    folder_path = PROJECT_ROOT / 'data' / 'processed'
    file_name = 'emails_cleaned.csv'
    full_path = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    emails_df.to_csv(full_path, index=False, encoding='utf-8')

    print(f"Saved successfully to: {full_path}")


if __name__ == "__main__":
    run_preprocessing()
