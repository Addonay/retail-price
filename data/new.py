# %% IMPORTS
import warnings

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# %% LOAD DATA
df = pl.read_csv("dataset.csv")
print("Dataset loaded successfully")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# %% DATA CLEANING
print(f"\nMissing values: {df.null_count().sum_horizontal().item()}")
print(f"Duplicates: {df.height - df.unique().height}")

df = df.fill_null(0)
print("Data cleaned - missing values filled with 0")

# %% CREATE TARGET VARIABLE
low_thresh = df.select(pl.col("qty").quantile(0.25)).item()
high_thresh = df.select(pl.col("qty").quantile(0.75)).item()

df = df.with_columns(
    pl.when(pl.col("qty") <= low_thresh)
    .then(0)
    .when(pl.col("qty") <= high_thresh)
    .then(1)
    .otherwise(2)
    .alias("demand_category")
)

print("\nDemand categories created:")
print(
    f"Low: <= {low_thresh:.1f}, Medium: {low_thresh:.1f}-{high_thresh:.1f}, High: > {high_thresh:.1f}"
)

# %% SELECT FEATURES
features = [
    "unit_price",
    "freight_price",
    "product_score",
    "customers",
    "weekday",
    "weekend",
    "holiday",
    "month",
    "comp_1",
    "comp_2",
]

X = df.select(features).to_numpy()
y = df.select("demand_category").to_numpy().flatten()

# %% TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %% DECISION TREE MODEL
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# %% K-NEAREST NEIGHBORS MODEL
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# %% ARTIFICIAL NEURAL NETWORK MODEL
ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
ann.fit(X_train, y_train)
ann_pred = ann.predict(X_test)
ann_accuracy = accuracy_score(y_test, ann_pred)

# %% MODEL COMPARISON
print("\n" + "=" * 50)
print("MODEL COMPARISON - TEST ACCURACIES")
print("=" * 50)
print(f"Decision Tree:    {dt_accuracy:.3f}")
print(f"K-NN:            {knn_accuracy:.3f}")
print(f"Neural Network:   {ann_accuracy:.3f}")
print("=" * 50)

# %% VISUALIZE DECISION TREE
plt.figure(figsize=(15, 10))
plot_tree(
    dt,
    feature_names=features,
    class_names=["Low", "Medium", "High"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3,
)
plt.title("Decision Tree (First 5 Features, Max Depth 3)")
plt.tight_layout()
plt.show()

# %% DETAILED RESULTS FOR BEST MODEL
best_model = (
    "Decision Tree"
    if dt_accuracy == max(dt_accuracy, knn_accuracy, ann_accuracy)
    else "K-NN"
    if knn_accuracy == max(dt_accuracy, knn_accuracy, ann_accuracy)
    else "Neural Network"
)

print(f"\nBest performing model: {best_model}")
print(f"\nDetailed Classification Report ({best_model}):")

if best_model == "Decision Tree":
    print(
        classification_report(
            y_test, dt_pred, target_names=["Low", "Medium", "High"]
        )
    )
elif best_model == "K-NN":
    print(
        classification_report(
            y_test, knn_pred, target_names=["Low", "Medium", "High"]
        )
    )
else:
    print(
        classification_report(
            y_test, ann_pred, target_names=["Low", "Medium", "High"]
        )
    )
