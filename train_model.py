import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import joblib


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("career_dataset.csv")

print("Dataset loaded successfully!")
print(f"Total records: {len(df)}")


# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = df.drop("career", axis=1)
y = df["career"]


# -----------------------------------------
# 3. Define categorical and numeric columns
# -----------------------------------------

categorical_features = ["branch"]

numeric_features = [
    "programming",
    "analytical_reasoning",
    "hardware",
    "mathematics",
    "communication",
    "data_structures",
    "operating_systems",
    "networking",
    "digital_electronics",
    "machine_learning"
]


# -----------------------------------------
# 4. Encode branch
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "branch",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# -----------------------------------------
# 5. Create Random Forest model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42
)


# -----------------------------------------
# 6. Create complete ML pipeline
# -----------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------------------
# 7. Split dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------------
# 8. Train model
# -----------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------------------
# 9. Make predictions
# -----------------------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------------------
# 10. Calculate accuracy
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n-----------------------------------------")
print("MODEL RESULTS")
print("-----------------------------------------")

print(f"Accuracy: {accuracy:.4f}")
print(
    f"Accuracy percentage: {accuracy * 100:.2f}%"
)


# -----------------------------------------
# 11. Classification report
# -----------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------------------
# 12. Confusion matrix
# -----------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=pipeline.classes_
)

print(cm)


# -----------------------------------------
# 13. ROC Curve
# -----------------------------------------

print("\nGenerating ROC Curve...")

# Get prediction probabilities
y_prob = pipeline.predict_proba(X_test)

# Get all career classes
classes = pipeline.classes_

# Convert actual labels into binary format
y_test_bin = label_binarize(
    y_test,
    classes=classes
)

# Create charts folder if it does not exist
import os

os.makedirs("charts", exist_ok=True)

plt.figure(figsize=(10, 8))

# Generate ROC curve for every career class
for i, class_name in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=1.5,
        label=f"{class_name} (AUC = {roc_auc:.2f})"
    )


# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random Classifier"
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Career Prediction Model"
)

plt.legend(
    loc="lower right",
    fontsize=7
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# Save ROC curve
plt.savefig(
    "charts/06_roc_curve.png",
    dpi=300
)

plt.close()

print("ROC curve generated successfully!")
print("File created: charts/06_roc_curve.png")


# -----------------------------------------
# 14. Save trained model
# -----------------------------------------

joblib.dump(
    pipeline,
    "career_model.pkl"
)

print("\nModel saved successfully!")
print("File created: career_model.pkl")