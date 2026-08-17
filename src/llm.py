from dotenv import load_dotenv
from groq import Groq
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def explain_prediction(subject, cleaned_subject, prediction, confidence):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Groq API key not found."

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
Explain why the following email subject is classified as {prediction}.

Subject: {subject}
Cleaned Subject: {cleaned_subject}
Prediction: {prediction.upper()}
Confidence: {confidence:.2%}

Give a short and simple explanation in max 2 sentences.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Explain spam detection briefly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=40,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        return "Failed to generate explanation."


explanation = explain_prediction(
    subject="Congratulations! You won a free iPhone. Click now!",
    cleaned_subject="congratulations won free iphone click",
    prediction="spam",
    confidence=0.9642,
)
print(explanation)

explanation = explain_prediction(
    subject="Meeting scheduled for tomorrow at 10 AM",
    cleaned_subject="meeting scheduled tomorrow 10 am",
    prediction="ham",
    confidence=0.9125,
)

print(explanation)
