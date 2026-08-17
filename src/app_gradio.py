"""
Gradio App for Email Subject Spam Filter

Interactive web application for classifying email subjects as spam or ham.
"""

import logging
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from typing import Tuple
from src.preprocessing import clean_subject
from src.llm import explain_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
MODEL = None
VECTORIZER = None
LABEL_ENCODER = None
TRAINING_DATA = None


def load_models(model_path: str, vectorizer_path: str, data_path: str = None):
    """
    Load trained model and vectorizer.

    Args:
        model_path: Path to saved model
        vectorizer_path: Path to saved vectorizer
        data_path: Path to training data (for similar examples)
    """
    global MODEL, VECTORIZER, TRAINING_DATA

    try:
        MODEL = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    try:
        VECTORIZER = joblib.load(vectorizer_path)
        logger.info(f"Loaded vectorizer from {vectorizer_path}")
    except Exception as e:
        logger.error(f"Error loading vectorizer: {e}")
        raise

    # Try to load training data
    if data_path:
        try:
            TRAINING_DATA = pd.read_csv(data_path)
            logger.info(f"Loaded training data from {data_path}")
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")


def predict_subject_spam(
    subject_text: str, include_explanation: bool = True
) -> Tuple[str, float, str, str]:
    """
    Predict if email subject is spam or ham.

    Args:
        subject_text: Email subject text
        include_explanation: Whether to include LLM explanation

    Returns:
        Tuple of (prediction, confidence, cleaned_subject, explanation)
    """
    if MODEL is None or VECTORIZER is None:
        return (
            "Error",
            0.0,
            "",
            "Model not loaded. Please ensure models are properly initialized.",
        )

    try:
        # Clean the subject
        cleaned_subject_text = clean_subject(subject_text)

        if not cleaned_subject_text:
            return (
                "Invalid",
                0.0,
                cleaned_subject_text,
                "Subject is empty after cleaning.",
            )

        # Transform using vectorizer
        X = VECTORIZER.transform([cleaned_subject_text])

        # Make prediction
        prediction = MODEL.predict(X)[0]

        # Get confidence
        confidence = 0.0
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[0]
            confidence = max(proba)
        elif hasattr(MODEL, "decision_function"):
            decision = MODEL.decision_function(X)[0]
            # Sigmoid to convert to probability
            from scipy.special import expit

            confidence = expit(decision)
        else:
            confidence = 0.5

        # Map prediction
        label = "Spam" if prediction == 1 else "Ham"

        # Generate explanation
        explanation = ""
        if include_explanation:
            explanation = explain_prediction(
                subject_text, cleaned_subject_text, label, confidence
            )

        return label, float(confidence), cleaned_subject_text, explanation

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return "Error", 0.0, "", f"Error during prediction: {str(e)}"


def find_similar_examples(subject_text: str, label: str, n_examples: int = 3) -> str:
    """
    Find similar examples from training data.

    Args:
        subject_text: Cleaned subject text
        label: Predicted label
        n_examples: Number of examples to return

    Returns:
        Formatted string with similar examples
    """
    if TRAINING_DATA is None or "subject_clean" not in TRAINING_DATA.columns:
        return "No training data available for similar examples."

    try:
        # Filter by label
        label_data = TRAINING_DATA[TRAINING_DATA["pseudo_label"] == label.lower()]

        if len(label_data) == 0:
            return f"No {label} examples found in training data."

        # Get random examples
        examples = label_data.sample(min(n_examples, len(label_data)))

        output = f"**Similar {label} Examples:**\n"
        for idx, row in examples.iterrows():
            output += f"- {row['subject_clean']}\n"

        return output

    except Exception as e:
        logger.error(f"Error finding similar examples: {e}")
        return "Could not find similar examples."


def create_gradio_interface():
    """
    Create Gradio interface for the spam filter.

    Returns:
        Gradio Interface object
    """

    def process_subject(
        subject: str, show_explanation: bool = True, show_similar: bool = False
    ) -> Tuple[str, str, str, str]:
        """Process subject and return results."""
        prediction, confidence, cleaned, explanation = predict_subject_spam(
            subject, include_explanation=show_explanation
        )

        # Format output
        result_text = f"""
### Prediction Result

**Email Subject:** {subject}

**Cleaned Subject:** {cleaned}

**Prediction:** {prediction}

**Confidence:** {confidence:.2%}

---
"""

        if show_explanation and explanation:
            result_text += f"""### Explanation

{explanation}

---
"""

        if show_similar and prediction != "Invalid" and prediction != "Error":
            similar = find_similar_examples(cleaned, prediction)
            result_text += f"\n{similar}"

        return prediction, f"{confidence:.2%}", cleaned, result_text

    with gr.Blocks(title="Email Subject Spam Filter") as demo:
        gr.Markdown("# Email Subject Spam Filter")
        gr.Markdown("""
This application classifies email subjects as **SPAM** or **HAM** using a machine learning model trained on the Enron email dataset.

**Note:** Labels are generated using a pretrained transformer model (pseudo-labels), not manually annotated ground truth.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                subject_input = gr.Textbox(
                    label="Email Subject",
                    placeholder="Enter an email subject line...",
                    lines=3,
                )

            with gr.Column(scale=1):
                show_explanation = gr.Checkbox(
                    label="Include LLM Explanation", value=True
                )
                show_similar = gr.Checkbox(label="Show Similar Examples", value=False)
                submit_btn = gr.Button("Classify", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                prediction_output = gr.Textbox(label="Prediction", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)

            with gr.Column(scale=1):
                cleaned_output = gr.Textbox(
                    label="Cleaned Subject", interactive=False, lines=3
                )

        with gr.Row():
            result_output = gr.Markdown(
                label="Detailed Results", value="Results will appear here..."
            )

        # Examples
        gr.Examples(
            examples=[
                ["Limited Time Offer! Buy Now and Save 50%"],
                ["Meeting Tomorrow at 10 AM"],
                ["URGENT: Confirm Your Password Now!!!"],
                ["Project Update - Q3 Results"],
                ["You Have Won $1,000,000!"],
                ["Lunch Plans Next Week"],
            ],
            inputs=subject_input,
        )

        # Connect button
        submit_btn.click(
            fn=process_subject,
            inputs=[subject_input, show_explanation, show_similar],
            outputs=[
                prediction_output,
                confidence_output,
                cleaned_output,
                result_output,
            ],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Gradio app for email subject spam filter"
    )
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--vectorizer", required=True, help="Path to saved vectorizer")
    parser.add_argument("--data", help="Path to training data")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    # Load models
    load_models(args.model, args.vectorizer, args.data)

    # Create and launch interface
    interface = create_gradio_interface()
    interface.launch(share=args.share, debug=True)
