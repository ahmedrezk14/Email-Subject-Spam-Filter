import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

pos_class = "spam"
neg_class = "ham"
labels_ = [neg_class, pos_class]
POS_LABEL = "spam"
NEG_LABEL = "ham"


def plot_top_words(texts, title, n=20, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=5000,
    )
    X_counts = vectorizer.fit_transform(texts.astype(str))
    counts = np.asarray(X_counts.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    top_df = pd.DataFrame({"term": terms, "count": counts})
    top_df = top_df.sort_values("count", ascending=False).head(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_df, y="term", x="count")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    return top_df


def get_positive_scores(model, X_values):
    """Get score for positive class. Used for ROC-AUC only."""
    classes = list(model.classes_)
    pos_index = classes.index(pos_class)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_values)
        return scores[:, pos_index], "proba"

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_values))
        if scores.ndim == 1:
            if classes[1] == pos_class:
                return scores, "decision"
            return -scores, "decision"
        return scores[:, pos_index], "decision"

    return None, None


def safe_roc_auc(y_true, scores):
    if scores is None:
        return np.nan
    try:
        y_true_binary = (pd.Series(y_true).values == pos_class).astype(int)
        return roc_auc_score(y_true_binary, scores)
    except Exception:
        return np.nan


def evaluate_predictions(y_true, y_pred, scores=None, prefix=""):
    return {
        prefix + "accuracy": accuracy_score(y_true, y_pred),
        prefix + "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        prefix + "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        prefix + "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        prefix + "spam_precision": precision_score(y_true, y_pred, pos_label=pos_class, zero_division=0),
        prefix + "spam_recall": recall_score(y_true, y_pred, pos_label=pos_class, zero_division=0),
        prefix + "spam_f1": f1_score(y_true, y_pred, pos_label=pos_class, zero_division=0),
        prefix + "roc_auc": safe_roc_auc(y_true, scores),
    }


def show_best_model_evaluation(results_df, trained_models, X_test, y_test):
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("Best model:", best_model_name)
    print("Best test spam F1:", results_df.iloc[0]["test_spam_f1"])
    print("Best test macro F1:", results_df.iloc[0]["test_f1_macro"])

    y_test_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_test_pred, labels=labels_, zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred, labels=labels_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.show()

    error_df = X_test.copy()
    error_df["true_label"] = y_test.values
    error_df["pred_label"] = y_test_pred
    error_df = error_df[error_df["true_label"] != error_df["pred_label"]]

    print("Misclassified samples:", len(error_df))
    print(error_df.head(20))

    return best_model_name, best_model, y_test_pred, error_df


def predict_from_threshold(scores, threshold):
    return np.where(scores >= threshold, POS_LABEL, NEG_LABEL)


def tune_threshold_svm(model, X_valid, y_valid):
    scores = model.decision_function(X_valid)
    thresholds = np.percentile(scores, np.arange(5, 96, 1))

    best_threshold = None
    best_f1 = -1
    best_precision = None
    best_recall = None

    for threshold in thresholds:
        preds = predict_from_threshold(scores, threshold)
        f1 = f1_score(y_valid, preds, pos_label=POS_LABEL, zero_division=0)
        precision = precision_score(y_valid, preds, pos_label=POS_LABEL, zero_division=0)
        recall = recall_score(y_valid, preds, pos_label=POS_LABEL, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_precision = precision
            best_recall = recall

    return best_threshold, best_f1, best_precision, best_recall
