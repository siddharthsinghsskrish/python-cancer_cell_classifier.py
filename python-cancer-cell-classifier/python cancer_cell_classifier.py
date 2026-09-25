# ============================================================
# Cancer Cell Classification using Machine Learning
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# ============================================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("        CANCER CELL CLASSIFICATION PROJECT")
print("=" * 60)

data = load_breast_cancer()

# Features
X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Target
y = data.target

print("\nDataset loaded successfully!")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nClasses:")
for number, name in enumerate(data.target_names):
    print(number, "=", name)


# ============================================================
# 2. DISPLAY DATA
# ============================================================

print("\nFirst 5 samples:")
print(X.head())


# ============================================================
# 3. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. CREATE MACHINE LEARNING MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. CHECK ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()

plt.title("Cancer Cell Classification - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\nTop 10 Important Features:")

print(importance.head(10))


# Plot top 10 features

plt.figure(figsize=(10, 6))

importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ============================================================
# 11. TEST ONE EXISTING SAMPLE
# ============================================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

predicted_class = data.target_names[prediction[0]]

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

print("Sample number: 1")
print("Predicted class:", predicted_class)


# ============================================================
# 12. SHOW PROBABILITY
# ============================================================

probability = model.predict_proba(sample)[0]

print("\nPrediction probabilities:")

for class_name, probability_value in zip(
    data.target_names,
    probability
):
    print(
        f"{class_name}: "
        f"{probability_value * 100:.2f}%"
    )


print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)