""" Machine Learning Internship
 Disease Prediction from Medical Data
--------------------------------------------
Objective: Predict the possibility of disease (breast cancer: malignant vs
benign) based on patient diagnostic measurements.

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset - built into
scikit-learn (sklearn.datasets.load_breast_cancer), so no manual download
is required. Features include things like radius, texture, perimeter,
area, smoothness, etc. of cell nuclei from breast mass biopsies.

Algorithms used: Logistic Regression, SVM, Random Forest
Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC

To swap in a different disease dataset (e.g. Heart Disease or Diabetes from
the UCI ML Repository), just replace `load_data()` with
`pd.read_csv("your_data.csv")` and set X / y accordingly - the rest of the
pipeline (scaling, training, evaluation, plots) stays the same.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------
def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # target: 0 = malignant, 1 = benign  (sklearn's default encoding)
    df.rename(columns={"target": "diagnosis"}, inplace=True)
    return df, data.target_names


# ---------------------------------------------------------------------
# 2. Main pipeline
# ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" DISEASE PREDICTION FROM MEDICAL DATA")
    print("(Breast Cancer Wisconsin Diagnostic Dataset)")
    print("=" * 60)

    df, target_names = load_data()
    df.to_csv("medical_data.csv", index=False)
    print(f"\nDataset loaded: {df.shape[0]} patients, {df.shape[1]-1} features")
    print(f"Classes: {list(target_names)} (0 = malignant, 1 = benign)")
    print(df.head())
    print("\nClass balance:")
    print(df["diagnosis"].value_counts())

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # Train multiple models (all use scaled features here, since
    # both SVM and Logistic Regression benefit from scaling)
    # ------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        fitted_models[name] = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc
        })

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print(f"ROC-AUC: {roc_auc:.4f}")

        # 5-fold cross-validation for a more robust estimate
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df)
    results_df.to_csv("model_comparison_results.csv")

    # ------------------------------------------------------------
    # Plot: metric comparison bar chart
    # ------------------------------------------------------------
    results_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Disease Prediction Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    print("\nSaved chart -> model_comparison.png")

    # ------------------------------------------------------------
    # Confusion matrix for best model (by ROC-AUC)
    # ------------------------------------------------------------
    best_model_name = results_df["ROC-AUC"].idxmax()
    best_model = fitted_models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print(f"Best model: {best_model_name} -> saved confusion_matrix.png")

    # ------------------------------------------------------------
    # Feature importance (Random Forest) - top 15
    # ------------------------------------------------------------
    rf_model = fitted_models["Random Forest"]
    importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False).head(15).sort_values()
    plt.figure(figsize=(8, 7))
    importance.plot(kind="barh", color="crimson")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved chart -> feature_importance.png")

    print("\nDone! All outputs saved in the current folder.")


if __name__ == "__main__":
    main()
