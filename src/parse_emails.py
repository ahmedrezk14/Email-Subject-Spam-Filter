"""
Email Parsing Module

This module provides functions to parse raw email messages and extract structured metadata.
Handles missing fields gracefully by filling with NaN/empty strings.
"""

import pandas as pd
import re
from email.parser import Parser
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_email_message(raw_message: str) -> Dict:
    """
    Parse a raw email message and extract key fields.

    Args:
        raw_message: Raw email text containing headers and body

    Returns:
        Dictionary with extracted fields
    """
    email_dict = {
        "message_id": "",
        "date": "",
        "from_email": "",
        "to_email": "",
        "cc": "",
        "bcc": "",
        "subject": "",
        "mime_version": "",
        "content_type": "",
        "content_transfer_encoding": "",
        "x_from": "",
        "x_to": "",
        "x_cc": "",
        "x_bcc": "",
        "x_folder": "",
        "x_origin": "",
        "x_filename": "",
        "body": "",
    }

    try:
        # Split headers from body
        if "\n\n" in raw_message:
            headers_part, body_part = raw_message.split("\n\n", 1)
        else:
            headers_part = raw_message
            body_part = ""

        # Parse headers using email parser
        parser = Parser()
        msg = parser.parsestr(headers_part)

        # Extract standard email headers
        email_dict["message_id"] = msg.get("Message-ID", "")
        email_dict["date"] = msg.get("Date", "")
        email_dict["from_email"] = msg.get("From", "")
        email_dict["to_email"] = msg.get("To", "")
        email_dict["cc"] = msg.get("Cc", "")
        email_dict["bcc"] = msg.get("Bcc", "")
        email_dict["subject"] = msg.get("Subject", "")
        email_dict["mime_version"] = msg.get("Mime-Version", "")
        email_dict["content_type"] = msg.get("Content-Type", "")
        email_dict["content_transfer_encoding"] = msg.get(
            "Content-Transfer-Encoding", ""
        )

        # Extract X-* headers (custom Enron headers)
        email_dict["x_from"] = msg.get("X-From", "")
        email_dict["x_to"] = msg.get("X-To", "")
        email_dict["x_cc"] = msg.get("X-cc", "")
        email_dict["x_bcc"] = msg.get("X-bcc", "")
        email_dict["x_folder"] = msg.get("X-Folder", "")
        email_dict["x_origin"] = msg.get("X-Origin", "")
        email_dict["x_filename"] = msg.get("X-FileName", "")

        # Extract body (everything after headers)
        email_dict["body"] = body_part.strip()

    except Exception as e:
        logger.warning(f"Error parsing email: {str(e)[:100]}")
        email_dict["body"] = raw_message  # If parsing fails, keep raw as body

    return email_dict


def parse_emails_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse a batch of raw email messages and create structured dataframe.

    Args:
        df: DataFrame with 'file' and 'message' columns

    Returns:
        DataFrame with parsed email fields
    """
    logger.info(f"Parsing {len(df)} emails...")

    # Parse each email
    parsed_emails = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            logger.info(f"Parsed {idx} / {len(df)} emails")

        parsed = parse_email_message(row["message"])
        parsed["file"] = row["file"]
        parsed_emails.append(parsed)

    # Create dataframe from parsed emails
    emails_df = pd.DataFrame(parsed_emails)

    # Reorder columns
    columns_order = [
        "file",
        "message_id",
        "date",
        "from_email",
        "to_email",
        "cc",
        "bcc",
        "subject",
        "mime_version",
        "content_type",
        "content_transfer_encoding",
        "x_from",
        "x_to",
        "x_cc",
        "x_bcc",
        "x_folder",
        "x_origin",
        "x_filename",
        "body",
    ]
    emails_df = emails_df[columns_order]

    logger.info(f"Parsing complete. Final shape: {emails_df.shape}")

    return emails_df


def get_email_statistics(df: pd.DataFrame) -> Dict:
    """
    Get basic statistics about parsed emails.

    Args:
        df: Parsed emails dataframe

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_emails": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "empty_subjects": (df["subject"] == "").sum() + df["subject"].isnull().sum(),
        "duplicated_subjects": df["subject"].duplicated().sum(),
        "emails_with_body": (df["body"].str.len() > 0).sum(),
    }
    return stats
