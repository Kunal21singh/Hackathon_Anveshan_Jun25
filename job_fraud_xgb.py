#!/usr/bin/env python3
"""Job Fraud Detection with XGBoost
-------------------------------------------------
Train an XGBoost model that classifies job postings
as genuine (0) or fraudulent (1) and outputs a
fake-probability percentage for every record.

Usage
-----
python job_fraud_xgb.py --train Train.csv --output predictions.csv

Dependencies
------------
pandas, scikit-learn, xgboost
Install via:
    pip install -U pandas scikit-learn xgboost
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import joblib  # for optional model persistence


def build_pipeline(text_col: str, num_cols: list, pos_weight: float) -> Pipeline:
    """Return a scikit-learn pipeline with TF-IDF + XGBoost."""
    text_pipe = TfidfVectorizer(
        max_features=40_000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    preprocess = ColumnTransformer([
        ("text", text_pipe, text_col),
        ("num",  "passthrough", num_cols)
    ])

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",          # switch to "gpu_hist" for CUDA
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=42
    )

    return Pipeline([
        ("prep", preprocess),
        ("clf", clf)
    ])


def main(args):
    # --- Load data -----------------------------------------------------------
    df = pd.read_csv(args.train)
    if "description" not in df.columns:
        raise ValueError("Column 'description' not found in input CSV.")

    # Concatenate or choose a text field; here we just use 'description'
    df["text"] = df["description"].fillna("")
    text_col = "text"
    num_cols = ["telecommuting", "has_company_logo", "has_questions"]
    target_col = "fraudulent"

    # Check presence of numeric fields
    for col in num_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in input CSV.")

    # Train-test split
    X = df[[text_col] + num_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle class imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = neg / pos if pos else 1.0

    # Build & fit model
    model = build_pipeline(text_col, num_cols, pos_weight)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n=== Evaluation on hold-out set ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("F1 score:", f1_score(y_test, y_pred))

    # Predict full data set
    proba = model.predict_proba(X)[:, 1]
    df["fake_percentage"] = (proba * 100).round(2)

    # Save predictions if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nPredictions with 'fake_percentage' saved to {args.output}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Explainability: SHAP + Word-Cloud
    # ─────────────────────────────────────────────────────────────────────────────
    def generate_explainability_artifacts(model, X_full,
                                        shap_path="assets/shap_summary.png",
                                        wc_path="assets/important_keywords.png",
                                        n_samples=100):
        """
        Create a SHAP summary plot and a word-cloud of top TF-IDF tokens.
        Saves files to the given paths (use "assets/" so Dash can auto-serve).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import shap
        from wordcloud import WordCloud

        # Ensure output folder exists
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)

        # ------------------ Prepare feature matrix exactly as XGBoost sees it ----
        prep = model.named_steps["prep"]
        vec_text = prep.named_transformers_["text"]        # TF-IDF vectorizer
        X_trans  = prep.transform(X_full)                  # sparse matrix

        # Construct feature-name list (text first, then numeric passthrough cols)
        text_features = vec_text.get_feature_names_out()   # 40 000 names
        num_features  = ["telecommuting", "has_company_logo", "has_questions"]
        feature_names = np.concatenate([text_features, num_features])

        # ------------------ SHAP summary plot (global importance) ----------------
        explainer  = shap.Explainer(model.named_steps["clf"])
        shap_vals  = explainer(X_trans[:n_samples])        # subsample for speed

        shap.summary_plot(shap_vals.values,
                        features=X_trans[:n_samples],
                        feature_names=feature_names,
                        show=False)
        plt.tight_layout()
        plt.savefig(shap_path, dpi=250)
        plt.clf()

        # ------------------ Word-cloud of token importances ----------------------
        importances = model.named_steps["clf"].feature_importances_
        word_scores = dict(zip(text_features,
                            importances[:len(text_features)]))  # only text part

        wc = WordCloud(width=800,
                    height=400,
                    background_color="white",
                    collocations=False).generate_from_frequencies(word_scores)
        wc.to_file(wc_path)

        print(f"✓ SHAP saved to {shap_path}\n✓ Word-cloud saved to {wc_path}")


    # ---------------------------------------------------------------------------
    # Call the helper (put this right after you calculate `proba`)
    # ---------------------------------------------------------------------------
    generate_explainability_artifacts(model, X)


    

    # Save model if requested
    if args.model_out:
        joblib.dump(model, args.model_out)
        print(f"Model serialized to {args.model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Fraud Detection with XGBoost")
    parser.add_argument("--train", required=True, help="Path to training CSV file")
    parser.add_argument("--output", help="Where to write CSV with predictions")
    parser.add_argument("--model-out", help="Optional path to save fitted model (.pkl)")
    args = parser.parse_args()
    main(args)
