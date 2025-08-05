# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                           confusion_matrix,
                           roc_auc_score,
                           roc_curve,
                           precision_score,
                           recall_score,
                           f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

## ----------------------------
## 1. DATA LOADING FROM DRIVE
## ----------------------------

# Initialize Spark
spark = SparkSession.builder \
    .appName("StockPrediction") \
    .master("local[*]") \
    .config("spark.driver.memory", "8G") \
    .getOrCreate()

# Load CSV from Drive (replace with your path)
file_path = "/content/drive/MyDrive/Stock_data.csv"  # Update this path
df = spark.read.csv(file_path, header=True, inferSchema=True)

## ----------------------------
## 2. DATA PREPROCESSING
## ----------------------------

# Clean and prepare data
processed_df = (df
    .withColumn("Date", to_date(col("Date"), "M/d/yyyy"))
    .withColumn("Open", col("Open").cast("double"))
    .withColumn("High", col("High").cast("double"))
    .withColumn("Low", col("Low").cast("double"))
    .withColumn("Close", col("Close").cast("double"))
    .withColumn("Volume", col("Volume").cast("long"))
    .na.fill(0)
    .orderBy("Date")
)

## ----------------------------
## 3. FEATURE ENGINEERING
## ----------------------------

window_spec = Window.orderBy("Date")

feature_df = (processed_df
    # Basic features
    .withColumn("price_change", col("Close") - col("Open"))
    .withColumn("daily_return",
               (col("Close") - lag("Close", 1).over(window_spec)) /
               lag("Close", 1).over(window_spec))
    .withColumn("volatility", col("High") - col("Low"))

    # Technical indicators
    .withColumn("ma_5", avg("Close").over(window_spec.rowsBetween(-4, 0)))
    .withColumn("ma_20", avg("Close").over(window_spec.rowsBetween(-19, 0)))

    # Temporal features
    .withColumn("day_of_week", dayofweek("Date"))
    .withColumn("month", month("Date"))

    # Target variable (1 if next day's close is higher)
    .withColumn("label",
               when(lead("Close", 1).over(window_spec) > col("Close"), 1).otherwise(0))

    # Filter out incomplete rows
    .filter(col("daily_return").isNotNull() & col("label").isNotNull())
)

## ----------------------------
## 4. MODEL TRAINING
## ----------------------------

# Convert to Pandas DataFrame
ml_data = feature_df.limit(10000).select([
    "price_change", "daily_return", "volatility",
    "ma_5", "ma_20", "Volume",
    "day_of_week", "month", "label"
]).toPandas()

# Train-test split
X = ml_data.drop("label", axis=1)
y = ml_data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

## ----------------------------
## 5. MODEL EVALUATION
## ----------------------------

# Generate predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Print metrics
print("\n=== MODEL PERFORMANCE ===\n")
print(classification_report(y_test, y_pred))
print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Down", "Up"],
            yticklabels=["Down", "Up"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
features = X.columns
importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

## ----------------------------
## 6. BUSINESS INTERPRETATION
## ----------------------------

print("\n=== BUSINESS INSIGHTS ===\n")

# Key metrics
precision_up = precision_score(y_test, y_pred, pos_label=1)
recall_down = recall_score(y_test, y_pred, pos_label=0)
accuracy = model.score(X_test, y_test)

print(f"Model Performance Summary:")
print(f"- Accuracy: {accuracy:.1%}")
print(f"- Precision (Up Moves): {precision_up:.1%}")
print(f"- Recall (Down Moves): {recall_down:.1%}")

print("\nTrading Strategy Recommendations:")
print("1. Strong Buy Conditions:")
print("   - When MA_5 > MA_20 and model confidence > 70%")
print("   - During high volume days (Volume > 1.5x average)")
print("2. Exit Signals:")
print("   - When price drops 2% below entry (stop-loss)")
print("   - After 3% profit or 3 trading days")

print("\nRisk Management Advice:")
print("- Reduce position size when volatility > 2.5x average")
print("- Avoid trading during holiday seasons (Nov-Jan)")
print("- Combine with news sentiment analysis for confirmation")