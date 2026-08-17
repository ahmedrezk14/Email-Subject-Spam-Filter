import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import joblib

from features import make_preprocess, numeric_cols, subject, target, pos_class, neg_class, labels_
from evaluate import (
    plot_top_words,
    get_positive_scores,
    evaluate_predictions,
    show_best_model_evaluation,
)

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

seed = 42
np.random.seed(seed)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_training(csv_path=None):
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "processed" / "emails_cleaned_labelled.csv"

    print("CSV path:", csv_path)
    df = pd.read_csv(csv_path)

    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    display(df.head())

    print("Data types:")
    display(df.dtypes.to_frame("dtype"))

    print("Missing values:")
    display(df.isna().sum().to_frame("missing"))

    print("Duplicate rows:", df.duplicated().sum())

    numeric_quality_cols = [
        "subject_char_len_after",
        "subject_word_count_after"
    ]

    for col in numeric_quality_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Outliers Check - {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    skewness_df = pd.DataFrame({
        "feature": numeric_quality_cols,
        "skewness": [df[col].skew() for col in numeric_quality_cols]
    })

    display(skewness_df)

    # Check required columns
    required_columns = [subject, target]

    # Important numeric features
    use_cols = [subject, target] + numeric_cols
    data = df[use_cols].copy()
    data = data.rename(columns={subject: "text", target: "label"})

    print("Selected numeric features:", numeric_cols)
    print("Before filtering:", data.shape)
    display(data.head())

    # Drop missing text/label rows
    data = data.dropna(subset=["text", "label"])

    # Normalize text and label
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.lower().str.strip()

    # Remove empty text
    data = data[data["text"] != ""]

    # Keep only ham/spam rows
    data = data[data["label"].isin(labels_)].copy()

    print("After basic filtering:", data.shape)
    print(data["label"].value_counts())

    display(data.head())

    print("Before removing duplicates")
    print("Rows:", len(data))
    print("Unique texts:", data["text"].nunique())
    print("Duplicate text rows:", data.duplicated(subset=["text"]).sum())
    print("Duplicate text + label rows:", data.duplicated(subset=["text", "label"]).sum())

    print("\nMissing values:")
    print(data.isna().sum().to_frame("missing"))

    # Remove duplicate text rows to avoid leakage between train and test
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("After removing duplicate text rows")
    print("Rows:", len(data))
    print("Unique texts:", data["text"].nunique())
    print("Duplicate text rows:", data.duplicated(subset=["text"]).sum())
    print("Duplicate text + label rows:", data.duplicated(subset=["text", "label"]).sum())

    # Label distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="label", order=labels_)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print(data["label"].value_counts())
    print(data["label"].value_counts(normalize=True))

    # Pie chart
    class_counts = data["label"].value_counts().reindex(labels_)

    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Spam vs Ham Percentage")
    plt.tight_layout()
    plt.show()

    # Text length distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data["subject_char_len_after"], bins=50, ax=axes[0])
    axes[0].set_title("Subject Character Length After Cleaning")
    axes[0].set_xlabel("Characters")

    sns.histplot(data["subject_word_count_after"], bins=50, ax=axes[1])
    axes[1].set_title("Subject Word Count After Cleaning")
    axes[1].set_xlabel("Words")

    plt.tight_layout()
    plt.show()

    corr_data = data[numeric_cols].copy()
    corr_data["label_encoded"] = data["label"].map({neg_class: 0, pos_class: 1})

    corr_matrix = corr_data.corr(numeric_only=True)

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Correlation Matrix - Numeric Features")
    plt.tight_layout()
    plt.show()

    top_unigrams = plot_top_words(data["text"], "Top 20 Words", n=20, ngram_range=(1, 1))
    display(top_unigrams)

    top_bigrams = plot_top_words(data["text"], "Top 20 Bigrams", n=20, ngram_range=(2, 2))
    display(top_bigrams)

    top_trigrams = plot_top_words(data["text"], "Top 20 Trigrams", n=20, ngram_range=(3, 3))
    display(top_trigrams)

    try:
        from wordcloud import WordCloud

        sample_text = data["text"].astype(str).sample(
            min(len(data), 50000), random_state=seed
        )
        all_text = " ".join(sample_text)

        wc = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            max_words=150,
        ).generate(all_text)

        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("WordCloud skipped:", e)

    X = data[["text"] + numeric_cols]
    y = data["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("\nSplit percentages:")
    print("Train:", round(len(X_train) / len(data) * 100, 2), "%")
    print("Validation:", round(len(X_val) / len(data) * 100, 2), "%")
    print("Test:", round(len(X_test) / len(data) * 100, 2), "%")

    split_dist = pd.DataFrame({
        "train": y_train.value_counts(normalize=True),
        "validation": y_val.value_counts(normalize=True),
        "test": y_test.value_counts(normalize=True),
    }).reindex(labels_).fillna(0)

    display(split_dist)

    split_dist.T.plot(kind="bar", figsize=(9, 5))
    plt.title("Class Distribution Across Splits")
    plt.ylabel("Ratio")
    plt.xlabel("Dataset Split")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("Feature pipeline is ready")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),

        "SVM": LinearSVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=5000,
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine",
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=30,
            class_weight="balanced",
            random_state=seed,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),

    }

    print("Models:")
    for name in models:
        print("-", name)

    param_grids = {
        "Logistic Regression": {
            "clf__C": [0.1, 1, 10],
        },

        "SVM": {
            "clf__C": [0.1, 1, 10],
        },

        "KNN": {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
        },

        "Decision Tree": {
            "clf__max_depth": [10, 20, 30],
            "clf__min_samples_split": [2, 5],
        },

        "Random Forest": {
            "clf__n_estimators": [100],
            "clf__max_depth": [20, 30],
        },

        "AdaBoost": {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.5, 0.8, 1.0],
        },
    }

    print("Evaluation helpers are ready")

    results = []
    trained_models = {}
    score_types = {}

    for model_name, model in models.items():
        print("=" * 90)
        print(f"Training: {model_name}")

        pipeline = Pipeline([
            ("features", make_preprocess()),
            ("clf", model),
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )

        start_train = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - start_train

        best_pipeline = grid.best_estimator_
        trained_models[model_name] = best_pipeline

        print("Best Params:", grid.best_params_)
        print("Best CV Score:", grid.best_score_)

        # Validation prediction
        start_pred = time.time()
        val_pred = best_pipeline.predict(X_val)
        val_pred_time = time.time() - start_pred

        # Test prediction
        start_pred = time.time()
        test_pred = best_pipeline.predict(X_test)
        test_pred_time = time.time() - start_pred

        # Scores for ROC-AUC only
        val_scores, score_type = get_positive_scores(best_pipeline, X_val)
        test_scores, _ = get_positive_scores(best_pipeline, X_test)

        val_metrics = evaluate_predictions(y_val, val_pred, scores=val_scores, prefix="val_")
        test_metrics = evaluate_predictions(y_test, test_pred, scores=test_scores, prefix="test_")

        row = {
            "model": model_name,
            "best_params": grid.best_params_,
            "best_cv_score": grid.best_score_,
            "train_time_sec": train_time,
            "val_prediction_time_sec": val_pred_time,
            "test_prediction_time_sec": test_pred_time,
            "score_type": score_type,
            **val_metrics,
            **test_metrics,
        }

        results.append(row)
        score_types[model_name] = score_type

        print(f"Done: {model_name}")
        print(f"Train time: {train_time:.2f} sec")
        print("Validation prediction distribution:")
        print(pd.Series(val_pred).value_counts())
        print("Test prediction distribution:")
        print(pd.Series(test_pred).value_counts())
        print(f"Validation spam F1: {val_metrics['val_spam_f1']:.4f}")
        print(f"Test spam F1: {test_metrics['test_spam_f1']:.4f}")
        print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")

    results_df = pd.DataFrame(results)

    # Sort by spam F1 first, then macro F1
    results_df = results_df.sort_values(
        by=["test_spam_f1", "test_f1_macro"],
        ascending=False,
    ).reset_index(drop=True)

    important_cols = [
        "model",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_spam_precision",
        "test_spam_recall",
        "test_spam_f1",
        "test_roc_auc",
        "train_time_sec",
        "test_prediction_time_sec",
        "best_params",
    ]

    display(results_df[important_cols])

    metrics_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)

    plot_df = results_df.copy()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="test_accuracy", y="model")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="test_spam_f1", y="model")
    plt.title("Model Spam F1 Comparison")
    plt.xlabel("Spam F1")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="test_f1_macro", y="model")
    plt.title("Model Macro F1 Comparison")
    plt.xlabel("Macro F1")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    best_model_name, best_model, y_test_pred, error_df = show_best_model_evaluation(
        results_df,
        trained_models,
        X_test,
        y_test,
    )

    best_model_path = OUTPUT_DIR / "best_model_pipeline.joblib"
    metadata_path = OUTPUT_DIR / "training_metadata.json"

    joblib.dump(best_model, best_model_path)

    metadata = {
        "best_model_name": best_model_name,
        "subject_column": subject,
        "target_column": target,
        "text_column_used": "text",
        "numeric_cols": numeric_cols,
        "labels": labels_,
        "positive_class": pos_class,
        "negative_class": neg_class,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Saved model:", best_model_path)
    print("Saved metadata:", metadata_path)

    from features import build_single_input

    def predict_subject_spam(subject_text):
        X_new = build_single_input(subject_text)
        pred = best_model.predict(X_new)[0]

        score, score_type = get_positive_scores(best_model, X_new)
        score_value = None if score is None else float(score[0])

        return {
            "original_subject": subject_text,
            "cleaned_subject": X_new.loc[0, "text"],
            "prediction": pred,
            "spam_score": score_value,
            "score_type": score_type,
        }

    examples = [
        "WIN $1000 NOW!!! Claim your free prize",
        "Meeting tomorrow about the project update",
        "Invoice attached for March payment",
        "Limited offer 50% discount click now",
    ]

    for ex in examples:
        print(predict_subject_spam(ex))

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "metadata": metadata,
        "model_path": best_model_path,
        "metadata_path": metadata_path,
    }


if __name__ == "__main__":
    run_training()
