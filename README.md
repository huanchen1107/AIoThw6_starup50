# Kaggle 50 Startups CRISP-DM Machine Learning Project (v2)

![Homework 6 Infographic Dashboard](./homework_infographic.jpg)

This repository implements a modular Scikit-learn regression solution to predict startup profitability based on expenditure profiles, adhering strictly to the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.

---

## 📂 Project Structure & Deliverables

- **[solve_50_startups_crispdm_v2.py](file:///d:/Huan/Chen/L6/solve_50_startups_crispdm_v2.py)**: The main CRISP-DM pipeline script containing modular functions representing Business Understanding, Data Understanding (EDA), Data Preparation (Pipeline with `ColumnTransformer` + `OneHotEncoder`), Modeling, Evaluation (5-Fold Cross Validation), and Deployment.
- **[data.csv](file:///d:/Huan/Chen/L6/data.csv)**: The primary 50 Startups dataset.
- **[startup_profit_model_v2.pkl](file:///d:/Huan/Chen/L6/startup_profit_model_v2.pkl)**: Serialized final pipeline model (refitted on all 50 samples using `R&D Spend` and `Marketing Spend`).
- **[feature_selection_performance.png](file:///d:/Huan/Chen/L6/feature_selection_performance.png)**: Visual showing the RMSE and R-squared curves for Forward Stepwise Selection, with a detailed features list table.
- **[feature_selection_performance_allinone.png](file:///d:/Huan/Chen/L6/feature_selection_performance_allinone.png)**: Comparative visual comparing 5 top feature selection algorithms (SFS, RFE, SelectKBest, Lasso, and Random Forest).
- **[hw6.md](file:///d:/Huan/Chen/L6/hw6.md)**: A concise homework summary report of today's work and results.

---

## 🚀 Execution & Running Instructions

Make sure the required libraries are installed:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### 1. Run the Main Machine Learning Pipeline
This command executes the full 6-step CRISP-DM pipeline, evaluates 4 feature-set models, prints selection justifications, and runs the deployment simulation:
```bash
python solve_50_startups_crispdm_v2.py
```

### 2. Generate the Dynamic Stepwise Feature Selection Plot
To regenerate the stepwise feature selection plot using `data.csv` (which can be replaced with your own custom data):
```bash
python "C:\Users\admin\.gemini\antigravity-ide\brain\87a539ea-1a2b-45e6-b9fd-b82a1818c06e\scratch\plot_results.py"
```

### 3. Generate the Unified 5-Algorithm Comparison Plot
To fit and compare the feature selection paths for SFS, RFE, SelectKBest, Lasso, and Random Forest:
```bash
python "C:\Users\admin\.gemini\antigravity-ide\brain\87a539ea-1a2b-45e6-b9fd-b82a1818c06e\scratch\plot_allinone.py"
```

---

## 📈 Summary of Results

### 1. Final Model Performance
The optimal model selected is **Model 2: R&D Spend + Marketing Spend**. Its test and cross-validated performance metrics are:
- **Test $R^2$ Score**: `0.947439` (Peaks here; adding more features degrades performance due to overfitting)
- **Test RMSE**: `$8,198.80`
- **5-Fold CV $R^2$ Mean**: `0.938885` ($\pm$ `0.037284`)
- **5-Fold CV RMSE Mean**: `$8,883.70`

### 2. Business Interpretation
$$\text{Profit} = \$46,975.86 + (0.7966 \times \text{R\&D Spend}) + (0.0299 \times \text{Marketing Spend})$$
- Each dollar spent on **R&D** is associated with an increase of **$\$0.80$** in net profit.
- Each dollar spent on **Marketing** is associated with an increase of **$\$0.03$** in net profit.
- Overhead costs (`Administration`) and location (`State`) do not show any significant independent association with Profit.

### 3. Deployed Simulation Prediction
For a new startup with $120,000 in R&D Spend and $250,000 in Marketing Spend, the deployed model predicts a profit of **`$150,042.92`**.
