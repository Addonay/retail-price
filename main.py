# %%
# # !uv init && uv venv && source .venv/bin/activate
# !uv sync

# %% IMPORTS
import warnings

import joblib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# %% LOAD DATA
df = pl.read_csv("data/dataset.csv")
print("Dataset loaded successfully")
print(f"Shape: {df.shape}")


# %% BASIC DATA INSPECTION
print("First 5 rows:")
df.head()

# %% CHECK FOR MISSING VALUES
null_counts = df.null_count()
print("Missing values per column:")
null_counts

# %% CHECK FOR DUPLICATES
duplicate_count = df.height - df.unique().height
print(f"Number of duplicate rows: {duplicate_count}")

# %% BASIC STATISTICS
print("Dataset info:")
print(f"Rows: {df.height}")
print(f"Columns: {df.width}")
print(f"Product categories: {df.select('product_category_name').n_unique()}")
print(f"Unique products: {df.select('product_id').n_unique()}")

# %% OUTLIER DETECTION - PRICE
price_stats = df.select(
    [
        pl.col("unit_price").quantile(0.25).alias("q1"),
        pl.col("unit_price").quantile(0.75).alias("q3"),
        pl.col("unit_price").median().alias("median"),
        pl.col("unit_price").mean().alias("mean"),
    ]
)
print("Unit price statistics:")
print(price_stats)

# %% OUTLIER DETECTION - QUANTITY
iqr_analysis = df.select(
    [
        (pl.col("qty").quantile(0.80) - pl.col("qty").quantile(0.20)).alias(
            "qty_iqr"
        ),
        pl.col("qty").quantile(0.20).alias("qty_q1"),
        pl.col("qty").quantile(0.80).alias("qty_q3"),
    ]
)
print("Quantity IQR analysis:")
print(iqr_analysis)

# %% VISUALIZE PRICE DISTRIBUTION
plt.figure(figsize=(10, 6))
price_data = df.select("unit_price").to_numpy().flatten()
plt.hist(price_data, bins=50, alpha=0.7, edgecolor="black")
plt.title("Unit Price Distribution")
plt.xlabel("Unit Price")
plt.ylabel("Frequency")
plt.show()

# %% VISUALIZE QUANTITY DISTRIBUTION
plt.figure(figsize=(10, 6))
qty_data = df.select("qty").to_numpy().flatten()
plt.hist(qty_data, bins=30, alpha=0.7, edgecolor="black")
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

# %% PRICE BY CATEGORY BOXPLOT
# Convert to format seaborn can use
plot_data = df.select(["product_category_name", "unit_price"]).to_pandas()
plt.figure(figsize=(12, 8))
sns.boxplot(x="product_category_name", y="unit_price", data=plot_data)
plt.title("Product Category vs. Unit Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% TOTAL PRICE BY CATEGORY
plot_data2 = df.select(["product_category_name", "total_price"]).to_pandas()
plt.figure(figsize=(12, 8))
sns.boxplot(x="product_category_name", y="total_price", data=plot_data2)
plt.title("Product Category vs. Total Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% COMPETITOR PRICE COMPARISON
comp_data = df.select(["unit_price", "comp_1", "comp_2", "comp_3"]).to_pandas()
plt.figure(figsize=(12, 6))
plt.scatter(
    comp_data["comp_1"], comp_data["unit_price"], alpha=0.6, label="vs Comp 1"
)
plt.scatter(
    comp_data["comp_2"], comp_data["unit_price"], alpha=0.6, label="vs Comp 2"
)
plt.xlabel("Competitor Price")
plt.ylabel("Our Price")
plt.title("Our Price vs Competitors")
plt.legend()
plt.show()

# %% SEASONAL ANALYSIS
seasonal_qty = df.group_by("month").agg(pl.col("qty").mean().alias("avg_qty"))
seasonal_data = seasonal_qty.to_pandas()
plt.figure(figsize=(10, 6))
plt.bar(seasonal_data["month"], seasonal_data["avg_qty"])
plt.title("Average Quantity by Month")
plt.xlabel("Month")
plt.ylabel("Average Quantity")
plt.show()

# %% CREATE TARGET VARIABLE - DEMAND CATEGORIES
# Create demand categories based on quantity quartiles
qty_quartiles = df.select(
    [
        pl.col("qty").quantile(0.25).alias("low_threshold"),
        pl.col("qty").quantile(0.75).alias("high_threshold"),
    ]
)

low_thresh = qty_quartiles.select("low_threshold").item()
high_thresh = qty_quartiles.select("high_threshold").item()

df = df.with_columns(
    [
        pl.when(pl.col("qty") <= low_thresh)
        .then(0)
        .when(pl.col("qty") <= high_thresh)
        .then(1)
        .otherwise(2)
        .alias("demand_category")
    ]
)

print(f"Demand thresholds: Low <= {low_thresh:.1f}, High > {high_thresh:.1f}")

# %% CHECK TARGET DISTRIBUTION
target_dist = df.group_by("demand_category").agg(pl.count().alias("count"))
print("Demand category distribution:")
print(target_dist)

# %% FEATURE ENGINEERING
df = df.with_columns(
    [
        # Price ratios
        (pl.col("unit_price") / pl.col("comp_1")).alias("price_ratio_comp1"),
        (pl.col("unit_price") / pl.col("comp_2")).alias("price_ratio_comp2"),
        # Price position vs competitors
        pl.when(pl.col("unit_price") < pl.col("comp_1"))
        .then(1)
        .otherwise(0)
        .alias("cheaper_than_comp1"),
        # Product characteristics
        (
            pl.col("product_name_lenght") / pl.col("product_description_lenght")
        ).alias("name_desc_ratio"),
        # Revenue per customer
        (pl.col("total_price") / pl.col("customers")).alias(
            "revenue_per_customer"
        ),
    ]
)

# %% SELECT FEATURES FOR MODELING
feature_cols = [
    "unit_price",
    "freight_price",
    "product_score",
    "customers",
    "weekday",
    "weekend",
    "holiday",
    "month",
    "volume",
    "comp_1",
    "comp_2",
    "comp_3",
    "product_weight_g",
    "product_photos_qty",
    "price_ratio_comp1",
    "price_ratio_comp2",
    "cheaper_than_comp1",
    "revenue_per_customer",
]

# %% PREPARE DATA FOR MODELING
model_data = df.select(feature_cols + ["demand_category"]).fill_null(0)
X = model_data.select(feature_cols).to_numpy()
y = model_data.select("demand_category").to_numpy().flatten()

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# %% TRAIN TEST SPLIT (50-50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# %% DECISION TREE HYPERPARAMETER TUNING
dt_params = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"],
}

dt = DecisionTreeClassifier(random_state=42)
# Use GridSearchCV instead of RandomizedSearchCV
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring="accuracy", n_jobs=-1)

# %% TRAIN DECISION TREE MODEL
# Fit the GridSearchCV to the training data
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

print(f"Best Decision Tree parameters: {dt_grid.best_params_}")
print(f"Best cross-validation score: {dt_grid.best_score_:.3f}")

# %% MAKE PREDICTIONS
y_pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.3f}")

# %% DETAILED CLASSIFICATION REPORT
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low Demand", "Medium Demand", "High Demand"],
    )
)

# %% CONFUSION MATRIX VISUALIZATION
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"],
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# %% FEATURE IMPORTANCE ANALYSIS
feature_importance = best_dt.feature_importances_
importance_df = pl.DataFrame(
    {"feature": feature_cols, "importance": feature_importance}
).sort("importance", descending=True)

print("Top 10 most important features:")
print(importance_df.head(10))

# %% VISUALIZE DECISION TREE
plt.figure(figsize=(20, 10))
plot_tree(
    best_dt,
    feature_names=feature_cols,
    class_names=["Low", "Medium", "High"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,
)
plt.title("Decision Tree Visualization (First 3 Levels)")
plt.show()

# %% SAVE MODEL
joblib.dump(best_dt, "data/model.pkl")

model_metadata = {
    "model_type": "DecisionTreeClassifier",
    "features": feature_cols,
    "best_params": dt_grid.best_params_,
    "test_accuracy": accuracy,
    "demand_thresholds": {"low": low_thresh, "high": high_thresh},
}

joblib.dump(model_metadata, "data/dt_metadata.pkl")
print("Decision tree model and metadata saved successfully!")


# %% PREDICTION FUNCTION
def predict_demand_category(model, features):
    """
    Predict demand category for new data
    """
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]

    categories = ["Low Demand", "Medium Demand", "High Demand"]
    return categories[prediction], probabilities


# %% SAMPLE PREDICTION
sample_features = X_test[0]
category, probs = predict_demand_category(best_dt, sample_features)
print(f"Sample prediction: {category}")
print(f"Probabilities: {probs}")
print(f"Actual: {['Low Demand', 'Medium Demand', 'High Demand'][y_test[0]]}")
