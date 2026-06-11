"""
CRISP-DM Methodology for 50 Startups Profit Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------------------------------------------
# Step 1: Business Understanding
# ---------------------------------------------------------
# Objective: Predict the profit of a startup based on its 
# spending (R&D, Administration, Marketing) and location (State).
# This helps venture capitalists or investors determine which 
# startups have the highest potential for profitability.

# =====================================
# CRISP-DM Step 2: Data Understanding
# 50 Startups Dataset
# =====================================

# -------------------------------------
# 1. Load Dataset
# -------------------------------------
df = pd.read_csv("data.csv")

print("="*60)
print("Dataset Shape")
print("="*60)
print(df.shape)

# -------------------------------------
# 2. First Look
# -------------------------------------
print("\n" + "="*60)
print("First 5 Records")
print("="*60)
print(df.head())

# -------------------------------------
# 3. Data Types
# -------------------------------------
print("\n" + "="*60)
print("Data Types")
print("="*60)
print(df.dtypes)

# -------------------------------------
# 4. Dataset Information
# -------------------------------------
print("\n" + "="*60)
print("Dataset Info")
print("="*60)
df.info()

# -------------------------------------
# 5. Missing Values
# -------------------------------------
print("\n" + "="*60)
print("Missing Values")
print("="*60)
print(df.isnull().sum())

# -------------------------------------
# 6. Duplicate Records
# -------------------------------------
print("\n" + "="*60)
print("Duplicate Records")
print("="*60)
print(df.duplicated().sum())

# -------------------------------------
# 7. Statistical Summary
# -------------------------------------
print("\n" + "="*60)
print("Descriptive Statistics")
print("="*60)
print(df.describe())

# =====================================
# 8. Numerical Feature Distribution
# =====================================
numeric_cols = [
    'R&D Spend',
    'Administration',
    'Marketing Spend',
    'Profit'
]

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# =====================================
# 9. Boxplot (Outlier Detection)
# =====================================
for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# =====================================
# 10. Correlation Matrix
# =====================================
corr_matrix = df.corr(numeric_only=True)

print("\n" + "="*60)
print("Correlation Matrix")
print("="*60)
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# =====================================
# 11. Scatter Plot (Feature vs Profit)
# =====================================
features = [
    'R&D Spend',
    'Administration',
    'Marketing Spend'
]

for feature in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=feature, y='Profit')
    plt.title(f'{feature} vs Profit')
    plt.show()

# =====================================
# 12. State Analysis
# =====================================
print("\n" + "="*60)
print("State Counts")
print("="*60)
print(df["State"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="State")
plt.title("State Distribution")
plt.show()

# =====================================
# 13. Profit by State
# =====================================
state_profit = (
    df.groupby("State")
      ["Profit"]
      .agg(["mean", "min", "max", "std"])
)

print("\n" + "="*60)
print("Profit Statistics by State")
print("="*60)
print(state_profit)

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="State", y="Profit")
plt.title("Profit by State")
plt.show()

# =====================================
# 14. Pairplot
# =====================================
sns.pairplot(df, hue="State")
plt.show()

# =====================================
# 15. Summary Findings (EDA Answers)
# =====================================
print("\n" + "="*60)
print("STEP 2 FINDINGS & EDA ANSWERS")
print("="*60)
print("""
### Data Quality
* Missing Values: None.
* Duplicate Records: None.

### Distribution
* Skewed: Administration is slightly skewed.
* Normal-like: Profit and R&D Spend are closer to normal distributions.

### Outliers
* Anomalies: Minimal outliers observed. A single outlier in Profit (lower end) might exist depending on the plot.
* Action: No aggressive removal is needed based on initial inspection.

### Correlation
* Most related to Profit: R&D Spend (High positive correlation).
* Least related to Profit: Administration (Very low correlation).

### State Analysis
* Highest avg Profit: New York and Florida slightly edge out, but variations are relatively small.
* State Impact: State doesn't show a massive direct impact on profit compared to R&D Spend.

### Preparation Candidates (Step 3)
| Feature         | Action                     |
| --------------- | -------------------------- |
| State           | One-Hot Encoding           |
| R&D Spend       | Keep                       |
| Marketing Spend | Keep                       |
| Administration  | Consider Feature Selection |
| Profit          | Target                     |
""")

# =====================================
# CRISP-DM Step 3: Data Preparation
# 50 Startups Dataset
# =====================================
print("\n" + "="*60)
print("CRISP-DM Step 3: Data Preparation")
print("="*60)

print("""
# 3.1 Objective
The objective of the Data Preparation phase is to transform the raw dataset into a machine-learning-ready format while preserving important business information discovered during the Data Understanding phase.
""")

# -------------------------------------
# 3.2 Feature and Target Separation
# -------------------------------------
print("\n" + "="*60)
print("3.2 Feature and Target Separation")
print("="*60)
X = df.drop("Profit", axis=1)
y = df["Profit"]
print("Features (X): R&D Spend, Administration, Marketing Spend, State")
print("Target (y): Profit")

# -------------------------------------
# 3.3 Handling Categorical Features
# -------------------------------------
print("\n" + "="*60)
print("3.3 Handling Categorical Features")
print("="*60)
print("Applying One-Hot Encoding to 'State' (dropping first to avoid dummy variable trap).")
categorical_cols = ["State"]

# -------------------------------------
# 3.4 Handling Numerical Features
# -------------------------------------
print("\n" + "="*60)
print("3.4 Handling Numerical Features")
print("="*60)
print("Applying StandardScaler to numerical features (Mean = 0, Std = 1).")
numerical_cols = ["R&D Spend", "Administration", "Marketing Spend"]

# -------------------------------------
# 3.5 Outlier Handling
# -------------------------------------
print("\n" + "="*60)
print("3.5 Outlier Handling")
print("="*60)
print("Keep all observations (50 records). Extreme values may represent legitimate business outcomes.")

# -------------------------------------
# 3.6 Feature Selection Strategy
# -------------------------------------
print("\n" + "="*60)
print("3.6 Feature Selection Strategy")
print("="*60)
print("Retain Administration in the baseline model. We may compare with/without it later.")

# -------------------------------------
# 3.7 Multicollinearity Investigation
# -------------------------------------
print("\n" + "="*60)
print("3.7 Multicollinearity Investigation")
print("="*60)
print("No feature is removed at this stage.")

# -------------------------------------
# 3.8 Train-Test Split
# -------------------------------------
print("\n" + "="*60)
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Training Set (80%): {X_train.shape[0]} records")
print(f"Testing Set (20%): {X_test.shape[0]} records")

# -------------------------------------
# 3.9 Preprocessing Pipeline Construction
# -------------------------------------
print("\n" + "="*60)
print("3.9 Preprocessing Pipeline Construction")
print("="*60)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        ),
        (
            "num",
            StandardScaler(),
            numerical_cols
        )
    ]
)
print("Pipeline constructed: OneHotEncoder -> StandardScaler")

# -------------------------------------
# 3.10 Final Prepared Dataset Summary
# -------------------------------------
print("\n" + "="*60)
print("3.10 Final Prepared Dataset Summary")
print("="*60)
print("""
| Task | Action |
|--------|--------|
| Missing Values | None Found |
| Duplicate Records | None Found |
| Target Selection | Profit |
| Feature Selection | R&D, Administration, Marketing, State |
| Categorical Handling | One-Hot Encoding |
| Numerical Scaling | StandardScaler |
| Outlier Handling | Retained |
| Feature Removal | Not Yet |
| Multicollinearity Check | Planned |
| Train/Test Split | 80/20 |
| Pipeline | ColumnTransformer |

Transitioning to Step 4 (Modeling)...
""")

# -------------------------------------
# 3.11 Feature Selection Analysis
# -------------------------------------
print("\n" + "="*60)
print("3.11 Feature Selection Analysis (5 Schemes)")
print("="*60)

# We need to fit the preprocessor on the training data to get transformed features
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)

# Get feature names after transformation
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
feature_names = list(cat_features) + numerical_cols

print(f"Transformed Feature Names: {feature_names}\n")

# 1. Correlation-based filtering
print("--- 1. Correlation with Profit ---")
df_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
df_transformed['Profit'] = y_train.reset_index(drop=True)
corr_with_profit = df_transformed.corr()['Profit'].drop('Profit').sort_values(ascending=False)
print(corr_with_profit)

# 2. SelectKBest
print("\n--- 2. SelectKBest (f_regression) ---")
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train_transformed, y_train)
for name, score in zip(feature_names, selector.scores_):
    print(f"{name}: F-score = {score:.2f}")

# 3. Recursive Feature Elimination (RFE)
print("\n--- 3. Recursive Feature Elimination (RFE) ---")
rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
rfe.fit(X_train_transformed, y_train)
for name, rank in zip(feature_names, rfe.ranking_):
    print(f"{name}: Rank {rank}")

# 4. Lasso Regularization
print("\n--- 4. Lasso Regularization ---")
lasso = Lasso(alpha=0.1) # Using a small alpha to observe shrinkage
lasso.fit(X_train_transformed, y_train)
for name, coef in zip(feature_names, lasso.coef_):
    print(f"{name}: Coef = {coef:.4f}")

# 5. Tree-based Feature Importance
print("\n--- 5. Tree-based Feature Importance ---")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_transformed, y_train)
for name, importance in zip(feature_names, rf.feature_importances_):
    print(f"{name}: Importance = {importance:.4f}")

print("\n(Note: Based on baseline requirements, we retain all features for the regression model below, but these insights can drive future optimizations.)")

# -------------------------------------
# 3.12 Backward Elimination Performance Plot
# -------------------------------------
print("\n" + "="*60)
print("3.12 Generating Feature Selection Plot")
print("="*60)

# Simulate Backward Elimination path
combos = [
    [0],             # R&D Spend
    [0, 2],          # R&D Spend + Marketing Spend
    [0, 2, 4],       # R&D Spend + Marketing Spend + State_New York
    [0, 2, 3, 4],    # R&D + Marketing + State_Florida + State_New York
    [0, 1, 2, 3, 4]  # All Features
]

rmses = []
r2s = []
ks = range(1, 6)
X_test_transformed = preprocessor.transform(X_test)

for combo in combos:
    X_train_k = X_train_transformed[:, combo]
    X_test_k = X_test_transformed[:, combo]
    
    model = LinearRegression()
    model.fit(X_train_k, y_train)
    y_pred = model.predict(X_test_k)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    rmses.append(rmse)
    r2s.append(r2)

# Create the plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ks, rmses, marker='o')
plt.title('RMSE by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.plot(ks, r2s, marker='o')
plt.title('R-squared by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')

plt.tight_layout()
plot_filename = "backward_elimination_plot.png"
plt.savefig(plot_filename)
print(f"Successfully generated and saved plot to {plot_filename}")

# ---------------------------------------------------------
# Step 4: Modeling
# ---------------------------------------------------------
print("\n--- Step 4: Modeling ---")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model_pipeline.fit(X_train, y_train)
print("Model training completed using Multiple Linear Regression.")

# ---------------------------------------------------------
# Step 5: Evaluation
# ---------------------------------------------------------
print("\n--- Step 5: Evaluation ---")
y_pred = model_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

results_df = pd.DataFrame({'Actual Profit': y_test, 'Predicted Profit': y_pred})
results_df['Difference'] = results_df['Actual Profit'] - results_df['Predicted Profit']
print("\nActual vs Predicted (Test Set):")
print(results_df.head(10))

# ---------------------------------------------------------
# Step 6: Deployment
# ---------------------------------------------------------
print("\n--- Step 6: Deployment ---")
model_filename = 'startup_profit_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"Model pipeline successfully saved to '{model_filename}'.")
print("Ready for deployment to production!")
