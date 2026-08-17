"""
Pseudo-Labeling Module

This module provides functions to generate pseudo-labels using a pretrained transformer model.
Uses Hugging Face transformers for spam/ham classification.
"""

import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_classifier(
    model_name: str = "text-classification", device: str = None
) -> pipeline:
    """
    Initialize a pretrained text classification pipeline.

    Args:
        model_name: Model identifier from Hugging Face
        device: Device to use ('cuda' for GPU, 'cpu' for CPU, None for auto)

    Returns:
        Hugging Face pipeline object
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    logger.info(f"Initializing classifier on device: {device}")
    logger.info(
        "Using model: distilbert-base-uncased-finetuned-sms-spam-detection (SMS Spam Collection)"
    )

    # Try using a model suitable for spam/ham detection
    # distilbert-base-uncased-finetuned-sms-spam-detection is trained on SMS spam data
    try:
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sms-spam-detection",
            device=device,
            top_k=None,
        )
    except Exception as e:
        logger.warning(f"Could not load specific model: {e}")
        logger.info("Falling back to default text-classification model")
        classifier = pipeline("text-classification", device=device)

    return classifier


def label_subjects_batch(
    texts: List[str], classifier: pipeline, batch_size: int = 32
) -> Tuple[List, List]:
    """
    Generate pseudo-labels for a batch of subject texts.

    Args:
        texts: List of subject texts
        classifier: Initialized classification pipeline
        batch_size: Number of texts to process at once

    Returns:
        Tuple of (predictions_list, confidences_list)
    """
    logger.info(
        f"Generating pseudo-labels for {len(texts)} subjects in batches of {batch_size}"
    )

    predictions = []
    confidences = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Pseudo-labeling"):
        batch = texts[i : i + batch_size]

        # Filter out empty texts
        batch_filtered = [
            (j, text)
            for j, text in enumerate(batch)
            if isinstance(text, str) and len(text.strip()) > 0
        ]

        if not batch_filtered:
            # All texts in batch are empty
            for _ in batch:
                predictions.append("ham")  # Default to ham
                confidences.append(0.5)
            continue

        try:
            # Get predictions
            batch_texts = [text for _, text in batch_filtered]
            batch_results = classifier(batch_texts)

            # Process results back to original batch order
            result_dict = {
                j: result
                for j, result in zip([idx for idx, _ in batch_filtered], batch_results)
            }

            for j in range(len(batch)):
                if j in result_dict:
                    result = result_dict[j]
                    if isinstance(result, list):
                        result = result[0]  # Take first result if multiple returned

                    label = result["label"].lower()
                    score = float(result["score"])

                    # Map POSITIVE/NEGATIVE to ham/spam
                    # Assume: POSITIVE = ham, NEGATIVE = spam (adjust as needed)
                    if "positive" in label or "negative" not in label:
                        pred_label = "ham"
                        confidence = score
                    else:
                        pred_label = "spam"
                        confidence = score

                    predictions.append(pred_label)
                    confidences.append(confidence)
                else:
                    # Text was empty
                    predictions.append("ham")
                    confidences.append(0.5)

        except Exception as e:
            logger.warning(f"Error processing batch: {e}")
            for _ in batch:
                predictions.append("ham")
                confidences.append(0.5)

    return predictions, confidences


def generate_pseudo_labels(
    df: pd.DataFrame,
    classifier: pipeline = None,
    confidence_threshold: float = 0.70,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Generate pseudo-labels for email subjects using pretrained transformer.

    Args:
        df: DataFrame with 'subject_clean' column
        classifier: Pretrained classifier pipeline (created if None)
        confidence_threshold: Minimum confidence to keep a label
        batch_size: Batch size for inference

    Returns:
        DataFrame with pseudo-labels and confidence scores, filtered by threshold
    """
    if classifier is None:
        classifier = initialize_classifier()

    logger.info(
        f"Generating pseudo-labels with confidence threshold >= {confidence_threshold}"
    )

    # Generate predictions
    predictions, confidences = label_subjects_batch(
        df["subject_clean"].tolist(), classifier, batch_size=batch_size
    )

    # Add to dataframe
    df["transformer_label"] = predictions
    df["transformer_confidence"] = confidences

    # Log distribution before filtering
    logger.info(f"\nLabel distribution before filtering:")
    logger.info(f"\n{df['transformer_label'].value_counts()}")

    initial_count = len(df)

    # Filter by confidence threshold
    df = df[df["transformer_confidence"] >= confidence_threshold].copy()
    filtered_count = initial_count - len(df)

    logger.info(
        f"\nFiltered {filtered_count} rows with confidence < {confidence_threshold}"
    )
    logger.info(f"Kept {len(df)} rows with confidence >= {confidence_threshold}")

    # Rename transformer_label to pseudo_label
    df.rename(columns={"transformer_label": "pseudo_label"}, inplace=True)

    # Log final statistics
    logger.info(f"\nFinal statistics:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nLabel distribution after filtering:")
    logger.info(f"\n{df['pseudo_label'].value_counts()}")
    logger.info(f"\nConfidence statistics:")
    logger.info(f"Mean: {df['transformer_confidence'].mean():.4f}")
    logger.info(f"Std: {df['transformer_confidence'].std():.4f}")
    logger.info(f"Min: {df['transformer_confidence'].min():.4f}")
    logger.info(f"Max: {df['transformer_confidence'].max():.4f}")

    return df.reset_index(drop=True)


def save_pseudo_labeled_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save pseudo-labeled data to CSV.

    Args:
        df: DataFrame with pseudo-labels
        filepath: Path to save CSV
    """
    # Keep only necessary columns
    save_df = df[["subject_clean", "pseudo_label", "transformer_confidence"]].copy()
    save_df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(save_df)} pseudo-labeled subjects to {filepath}")
