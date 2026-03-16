import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import cross_val_score

# ===============================
# Load Model Artifact
# ===============================

artifact = joblib.load("model.pkl")

models = artifact["models"]
scaler = artifact["scaler"]
FEATURE_NAMES = artifact["features"]

TARGET = "target"

# ===============================
# Create results folder
# ===============================

os.makedirs("results", exist_ok=True)

# ===============================
# Load dataset
# ===============================

df = pd.read_csv("heart.csv")

X = df[FEATURE_NAMES]
y = df[TARGET]

X_scaled = scaler.transform(X)

# ===============================
# Train Test Split (same seed)
# ===============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

# ===============================
# Store evaluation results
# ===============================

results = []

# ===============================
# ROC Curve Setup
# ===============================

plt.figure(figsize=(8,6))

# ===============================
# Evaluate each model
# ===============================

for name, model in models.items():

    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
    else:
        probs = preds

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    specificity = tn / (tn + fp)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        specificity
    ])

    # Print classification report
    print("\n======================")
    print(name)
    print("======================")
    print(classification_report(y_test, preds))

    # ===============================
    # Confusion Matrix
    # ===============================

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"results/confusion_matrix_{name}.png")
    plt.close()

    # ===============================
    # ROC Curve
    # ===============================

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

# ===============================
# Final ROC Plot
# ===============================

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()

plt.savefig("results/roc_curve.png")
plt.close()

# ===============================
# Results Table
# ===============================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "Specificity"
    ]
)

print("\nModel Performance Table")
print(results_df)

results_df.to_csv("results/model_performance.csv", index=False)

# ===============================
# Accuracy Comparison Graph
# ===============================

plt.figure(figsize=(7,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results_df
)

plt.title("Model Accuracy Comparison")

plt.savefig("results/model_accuracy_comparison.png")
plt.close()

# ===============================
# Feature Importance (Random Forest)
# ===============================

if "Random Forest" in models:

    rf = models["Random Forest"]

    importances = rf.feature_importances_

    feat_imp = pd.Series(importances, index=FEATURE_NAMES)

    feat_imp = feat_imp.sort_values(ascending=False)

    plt.figure(figsize=(8,5))

    sns.barplot(
        x=feat_imp,
        y=feat_imp.index
    )

    plt.title("Feature Importance - Random Forest")

    plt.savefig("results/feature_importance.png")
    plt.close()

# ===============================
# Correlation Heatmap
# ===============================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.savefig("results/correlation_heatmap.png")
plt.close()

# ===============================
# Cross Validation
# ===============================

print("\nCross Validation Results")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5
    )

    print(f"{name} CV Accuracy: {scores.mean():.4f}")