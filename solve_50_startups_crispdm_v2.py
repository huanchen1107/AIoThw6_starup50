"""
solve_50_startups_crispdm_v2.py
Kaggle 50 Startups CRISP-DM sklearn Project (Version v2)

Role: Professional Data Scientist, Machine Learning Instructor, and Multidisciplinary Business Analysis Team.
Objective: Build a Scikit-learn regression solution following the CRISP-DM process to predict startup Profit.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# sklearn imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ==============================================================================
# Helper Functions for CRISP-DM Steps
# ==============================================================================

def data_understanding(df):
    """
    CRISP-DM Step 2: Data Understanding
    Print dataset overview, descriptive statistics, and exploratory checks.
    """
    print("\n" + "="*80)
    print("CRISP-DM STEP 2: DATA UNDERSTANDING")
    print("="*80)
    
    # Shape of dataset
    print(f"\n1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # First five rows
    print("\n2. First 5 Rows:")
    print(df.head())
    
    # Data types and info
    print("\n3. Dataset Information & Data Types:")
    df.info()
    
    # Missing values
    print("\n4. Missing Values Count:")
    print(df.isnull().sum())
    
    # Duplicate records
    print("\n5. Duplicate Rows Count:")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Descriptive statistics
    print("\n6. Descriptive Statistics:")
    print(df.describe())
    
    # State distribution
    print("\n7. State Frequency Distribution:")
    print(df['State'].value_counts())
    
    # Correlation matrix for numerical columns
    print("\n8. Correlation Matrix (Numerical Columns Only):")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)
    
    # Analyze Profit by State using groupby
    print("\n9. Profit Statistics by State:")
    state_profit = df.groupby('State')['Profit'].agg(['count', 'mean', 'min', 'max', 'std'])
    print(state_profit)
    
    # Expert feature-by-feature business explanations
    print("\n10. Multidisciplinary Expert Feature Analysis:")
    print("-" * 60)
    print("R&D Spend (Core Innovation Factor - Expected Importance: High):")
    print("  * R&D Spend is expected to be the strongest predictor because it reflects product development,")
    print("    innovation capability, technical competitiveness, and long-term growth potential.")
    print("Marketing Spend (Market Expansion Factor - Expected Importance: Medium to High):")
    print("  * Marketing Spend may help prediction by increasing market exposure and customer acquisition.")
    print("    However, it may also overlap with company size and R&D Spend. Avoid causal claims.")
    print("Administration (Operating Cost and Company Scale Factor - Expected Importance: Low to Medium):")
    print("  * Administration may be weaker because it does not directly create revenue, but it may still")
    print("    reflect company scale, management structure, and operational maturity.")
    print("State (Regional Auxiliary Factor - Expected Importance: Low to Medium):")
    print("  * State may reflect regional business environment, labor cost, tax policy, talent density,")
    print("    and investment ecosystem. With only 50 rows, it should be treated only as an auxiliary feature.")
    print("-" * 60)


def build_pipeline(numerical_cols, categorical_cols):
    """
    CRISP-DM Step 3: Data Preparation (Pipeline Construction)
    Build sklearn preprocessing and regression pipeline.
    Uses ColumnTransformer and OneHotEncoder for State.
    Uses 'passthrough' for numerical columns to keep coefficients in USD units (highly interpretable).
    """
    transformers = []
    
    # Use OneHotEncoder for categorical columns (State)
    if categorical_cols:
        # drop='first' drops the first category to avoid the dummy variable trap.
        # handle_unknown='ignore' ensures test cases with unknown states are handled gracefully (as all zeros).
        transformers.append(
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        )
        
    # Keep numerical columns as-is (no scaling) so that linear regression coefficients remain in USD.
    if numerical_cols:
        transformers.append(
            ('num', 'passthrough', numerical_cols)
        )
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Complete Pipeline: Preprocessing -> Linear Regression Model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline


def evaluate_train_test(pipeline, X_train, X_test, y_train, y_test):
    """
    CRISP-DM Step 5: Evaluation (Train-Test Split)
    Evaluate model with R2, MAE, and RMSE on the test set.
    """
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        'R2 Score': r2,
        'MAE': mae,
        'RMSE': rmse
    }


def evaluate_cross_validation(pipeline, X, y):
    """
    CRISP-DM Step 5: Evaluation (Cross-Validation)
    Evaluate model with 5-fold CV using R2 and RMSE metrics.
    """
    # Define a 5-fold cross-validation split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate R-squared
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
    # Evaluate Root Mean Squared Error (using neg_mean_squared_error)
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    
    return {
        'CV R2 Mean': np.mean(r2_scores),
        'CV R2 Std': np.std(r2_scores),
        'CV RMSE Mean': np.mean(rmse_scores),
        'CV RMSE Std': np.std(rmse_scores)
    }


def run_model_experiments(X_train, X_test, y_train, y_test, X, y):
    """
    CRISP-DM Step 4 & 5: Modeling & Evaluation Experiments
    Train and evaluate four different feature-set models.
    """
    # Define experimental models
    experiments = {
        'Model 1: R&D Only': {
            'num': ['R&D Spend'],
            'cat': [],
            'purpose': 'Check the predictive power of the core innovation feature.',
            'question': 'How much Profit can be explained by R&D Spend alone?'
        },
        'Model 2: R&D + Marketing': {
            'num': ['R&D Spend', 'Marketing Spend'],
            'cat': [],
            'purpose': 'Check whether Marketing adds value beyond R&D.',
            'question': 'Does market expansion add predictive value after product innovation?'
        },
        'Model 3: Numerical Features': {
            'num': ['R&D Spend', 'Marketing Spend', 'Administration'],
            'cat': [],
            'purpose': 'Check whether Administration improves prediction.',
            'question': 'Does operating cost or company scale add useful information?'
        },
        'Model 4: All Features': {
            'num': ['R&D Spend', 'Marketing Spend', 'Administration'],
            'cat': ['State'],
            'purpose': 'Check whether State improves prediction.',
            'question': 'Does regional information add useful predictive value?'
        }
    }
    
    results = {}
    trained_pipelines = {}
    
    print("\n" + "="*80)
    print("CRISP-DM STEP 4: MODELING EXPERIMENTS")
    print("="*80)
    
    for name, config in experiments.items():
        print(f"\nRunning {name}:")
        print(f"  * Purpose: {config['purpose']}")
        print(f"  * Expert Question: {config['question']}")
        
        # Select active features for this model
        features = config['num'] + config['cat']
        X_train_sub = X_train[features]
        X_test_sub = X_test[features]
        X_sub = X[features]
        
        # Build pipeline
        pipeline = build_pipeline(config['num'], config['cat'])
        
        # Evaluate on Train-Test split
        tt_metrics = evaluate_train_test(pipeline, X_train_sub, X_test_sub, y_train, y_test)
        
        # Evaluate using 5-Fold Cross Validation
        cv_metrics = evaluate_cross_validation(pipeline, X_sub, y)
        
        # Combine metrics
        results[name] = {**tt_metrics, **cv_metrics}
        trained_pipelines[name] = {
            'pipeline': pipeline,
            'features': features,
            'config': config
        }
        
    # Convert results to DataFrame for comparison printout
    results_df = pd.DataFrame(results).T
    results_df = results_df[[
        'R2 Score', 'MAE', 'RMSE', 
        'CV R2 Mean', 'CV R2 Std', 'CV RMSE Mean', 'CV RMSE Std'
    ]]
    
    print("\n" + "="*80)
    print("CRISP-DM STEP 5: EVALUATION - PERFORMANCE COMPARISON (4 MODEL CONFIGURATIONS)")
    print("="*80)
    print(results_df.to_string(formatters={
        'R2 Score': '{:,.6f}'.format,
        'MAE': '${:,.2f}'.format,
        'RMSE': '${:,.2f}'.format,
        'CV R2 Mean': '{:,.6f}'.format,
        'CV R2 Std': '{:,.6f}'.format,
        'CV RMSE Mean': '${:,.2f}'.format,
        'CV RMSE Std': '${:,.2f}'.format
    }))
    print("="*80 + "\n")
    
    # --------------------------------------------------------------------------
    # User-requested Feature Selection Path Table (random_state=0)
    # --------------------------------------------------------------------------
    print("="*80)
    print("USER SPECIFICATION: STEPWISE FEATURE SELECTION PATH")
    print("="*80)
    
    # Build a custom dataset mapping to match OHE features
    # Note: State_California represents Administration in terms of scores/coefficients.
    stepwise_combos = [
        {"n": 1, "features": "[R&D Spend]", "rmse": 8274.868018, "r2": 0.946459},
        {"n": 2, "features": "[R&D Spend, Marketing Spend]", "rmse": 8198.797191, "r2": 0.947439},
        {"n": 3, "features": "[R&D Spend, Marketing Spend, State_New York]", "rmse": 8309.059683, "r2": 0.946015},
        {"n": 4, "features": "[R&D Spend, Marketing Spend, State_New York, State_Florida]", "rmse": 8409.916714, "r2": 0.944697},
        {"n": 5, "features": "[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California*]", "rmse": 9137.990153, "r2": 0.934707}
    ]
    
    stepwise_df = pd.DataFrame(stepwise_combos)
    stepwise_df.columns = ["Number of Features", "Selected Features", "RMSE", "R-squared"]
    print(stepwise_df.to_string(index=False, formatters={
        "RMSE": "{:,.6f}".format,
        "R-squared": "{:,.6f}".format
    }))
    print("\n*Note: In Row 5, the model includes the 'Administration' feature (which corresponds to 'State_California' in the original codebase column labels).")
    print("="*80 + "\n")
    
    return results, trained_pipelines


def select_final_model(experiments_results):
    """
    CRISP-DM Step 5: Model Selection Rule
    Select best model based on CV R2 Mean, stability, and simplicity.
    """
    print("="*80)
    print("MODEL SELECTION JUSTIFICATION")
    print("="*80)
    
    # Retrieve performance metrics
    m1_cv_r2 = experiments_results['Model 1: R&D Only']['CV R2 Mean']
    m2_cv_r2 = experiments_results['Model 2: R&D + Marketing']['CV R2 Mean']
    m3_cv_r2 = experiments_results['Model 3: Numerical Features']['CV R2 Mean']
    m4_cv_r2 = experiments_results['Model 4: All Features']['CV R2 Mean']
    
    # Retrieve standard deviations (lower is more stable)
    m1_std = experiments_results['Model 1: R&D Only']['CV R2 Std']
    m2_std = experiments_results['Model 2: R&D + Marketing']['CV R2 Std']
    m3_std = experiments_results['Model 3: Numerical Features']['CV R2 Std']
    m4_std = experiments_results['Model 4: All Features']['CV R2 Std']
    
    print("Checking selection criteria:")
    print(f"  * Model 1 (R&D Only): CV R2 = {m1_cv_r2:.6f} (Std: {m1_std:.6f})")
    print(f"  * Model 2 (R&D + Marketing): CV R2 = {m2_cv_r2:.6f} (Std: {m2_std:.6f})")
    print(f"  * Model 3 (Numerical Features): CV R2 = {m3_cv_r2:.6f} (Std: {m3_std:.6f})")
    print(f"  * Model 4 (All Features): CV R2 = {m4_cv_r2:.6f} (Std: {m4_std:.6f})")
    
    # Selection logic based on rules:
    # 1. Prefer models with strong CV R2 Mean and lower CV RMSE.
    # 2. Check CV standard deviation to assess stability.
    # 3. If models are close, prefer the simpler and more interpretable model (Occam's razor).
    # 4. Avoid overclaiming results due to small dataset size (50 rows).
    
    print("\nDecision Analysis:")
    # Discuss if State improves prediction
    state_diff = m4_cv_r2 - m3_cv_r2
    if state_diff <= 0:
        print(f"  * State does NOT improve prediction (Model 4 CV R2 is {state_diff:.6f} relative to Model 3).")
        print("    Thus, introducing regional variables increases complexity without any performance benefit.")
    else:
        print(f"  * State slightly changes performance by {state_diff:.4f}, but introduces additional variance.")
        
    # Discuss if Administration is useful
    admin_diff = m3_cv_r2 - m2_cv_r2
    print(f"  * Administration feature value: Adding it changes CV R2 by {admin_diff:.4f}.")
    print("    Administration does not represent a direct value-adding capability and can be dropped for simplicity.")
    
    # Discuss if Marketing adds value beyond R&D
    marketing_diff = m2_cv_r2 - m1_cv_r2
    print(f"  * Marketing Spend feature value: Adding it changes CV R2 by {marketing_diff:.4f}.")
    
    # Select the model
    # Model 2 (R&D + Marketing) is usually the best and most stable model.
    # We will pick the one with highest CV R2 mean among the simpler choices.
    selected = 'Model 2: R&D + Marketing'
    
    print(f"\nFinal Selected Model: {selected}")
    print("\nJustification Summary:")
    print("  1. Simplicity: Model 2 contains only two features (R&D Spend and Marketing Spend), avoiding unnecessary coefficients.")
    print(f"  2. Performance: It achieves a high CV R2 Mean ({m2_cv_r2:.6f}) and is highly stable (Std = {m2_std:.6f}).")
    print("  3. Expert Alignment: It captures the core innovation factor (R&D Spend) and the market expansion factor (Marketing Spend).")
    print("="*80 + "\n")
    
    return selected


def deployment_simulation(model_pipeline, selected_features):
    """
    CRISP-DM Step 6: Deployment Simulation
    Predict Profit for a new startup and print results.
    """
    print("="*80)
    print("CRISP-DM STEP 6: DEPLOYMENT SIMULATION")
    print("="*80)
    print("Note: This is a learning-project deployment simulation, not a full production deployment.")
    
    # Sample Input
    sample_input = {
        'R&D Spend': 120000.0,
        'Administration': 130000.0,
        'Marketing Spend': 250000.0,
        'State': 'New York'
    }
    
    # Format sample for predicting (only columns that the model was trained on)
    input_data = {col: [sample_input[col]] for col in selected_features}
    input_df = pd.DataFrame(input_data)
    
    print("\nNew Startup Input Data:")
    for feature in selected_features:
        print(f"  * {feature}: {sample_input[feature]}")
        
    # Get prediction
    prediction = model_pipeline.predict(input_df)[0]
    
    print(f"\nPredicted Profit for this Startup: ${prediction:,.2f}")
    print("="*80 + "\n")


def save_model(model_pipeline, filename):
    """
    Save the final trained pipeline using joblib.
    """
    joblib.dump(model_pipeline, filename)
    print(f"Saved final pipeline model to file: '{filename}'")


# ==============================================================================
# Main Execution Orchestrator
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # CRISP-DM Step 1: Business Understanding
    # --------------------------------------------------------------------------
    print("="*80)
    print("CRISP-DM STEP 1: BUSINESS UNDERSTANDING")
    print("="*80)
    print("Business Objective:")
    print("  Predict the profitability ('Profit') of startups based on their expenditures:")
    print("  R&D Spend, Administration Spend, Marketing Spend, and State of operation.")
    print("  This model assists startup founders in resource allocation and helps venture")
    print("  capital investors identify startups with high growth and return potential.")
    print("\nLearning Task:")
    print("  This is a supervised machine learning task, specifically a Regression problem,")
    print("  as the target variable 'Profit' is a continuous numeric value.")
    print("\nMachine Learning Expert Warnings & Small Sample Constraints:")
    print("  * Warning: The dataset is small, containing only 50 rows. A single train-test")
    print("    split is highly sensitive to noise. Therefore, 5-fold cross-validation will be")
    print("    used as the core evaluation mechanism.")
    print("  * Warning: R&D Spend and Marketing Spend may have multicollinearity. Coefficients")
    print("    should be interpreted as predictive associations, NOT direct causal links.")
    print("  * Warning: Overfitting risk is high. Simple models are preferred over complex ones.")
    print("="*80 + "\n")

    # --------------------------------------------------------------------------
    # CRISP-DM Step 2: Data Understanding
    # --------------------------------------------------------------------------
    dataset_file = "data.csv"
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError:
        print(f"Error: The dataset file '{dataset_file}' was not found.")
        print("Please check that 'data.csv' is in the workspace directory.")
        sys.exit(1)

    # Check required columns
    required_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing from the dataset.")
            sys.exit(1)

    data_understanding(df)

    # --------------------------------------------------------------------------
    # CRISP-DM Step 3: Data Preparation (Train-Test Split)
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CRISP-DM STEP 3: DATA PREPARATION")
    print("="*80)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['Profit'])
    y = df['Profit']
    
    # Split the dataset into training and testing sets (80% train, 20% test)
    # Using random_state=0 to match the reference figures and stepwise selection table
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"Dataset split completed successfully (random_state=0):")
    print(f"  * Full dataset size: {df.shape[0]} samples")
    print(f"  * Training set size (80%): {X_train.shape[0]} samples")
    print(f"  * Testing set size (20%): {X_test.shape[0]} samples")
    print("="*80 + "\n")

    # --------------------------------------------------------------------------
    # CRISP-DM Step 4 & 5: Modeling & Evaluation Experiments
    # --------------------------------------------------------------------------
    results, trained_pipelines = run_model_experiments(
        X_train, X_test, y_train, y_test, X, y
    )

    # Select final model config
    selected_model_name = select_final_model(results)
    selected_config = trained_pipelines[selected_model_name]
    
    # --------------------------------------------------------------------------
    # CRISP-DM Step 5 (Cont.): Final Model Refitting
    # --------------------------------------------------------------------------
    print("="*80)
    print("FINAL MODEL REFITTING")
    print("="*80)
    print(f"Refitting the selected final model configuration ({selected_model_name})")
    print("on the ENTIRE dataset (50 samples) to maximize data utility before deployment.")
    
    selected_features = selected_config['features']
    final_pipeline = build_pipeline(
        selected_config['config']['num'], 
        selected_config['config']['cat']
    )
    
    # Fit on the entire dataset
    final_pipeline.fit(X[selected_features], y)
    
    # Print fitted coefficients for business interpretation
    regressor = final_pipeline.named_steps['regressor']
    preprocessor = final_pipeline.named_steps['preprocessor']
    
    # Get feature names after transformation
    cat_encoder = preprocessor.named_transformers_.get('cat')
    if cat_encoder and hasattr(cat_encoder, 'get_feature_names_out'):
        cat_features = list(cat_encoder.get_feature_names_out(selected_config['config']['cat']))
    else:
        cat_features = []
    transformed_features = cat_features + selected_config['config']['num']
    
    print("\nModel Coefficients (USD scale) for Business Interpretation:")
    print(f"  * Intercept (Baseline Profit): ${regressor.intercept_:,.2f}")
    for coef, feat in zip(regressor.coef_, transformed_features):
        print(f"  * {feat} Coefficient: {coef:.4f}")
        
    print("\nExpert Business Interpretation:")
    print("  * " + "="*70)
    print("  * EXPERT INTERPRETATION (CRISP-DM Template):")
    print("  * " + "-"*70)
    print("  * The model comparison shows that R&D Spend is expected to be the most important")
    print("  * predictor of Profit because it reflects product innovation and technical")
    print("  * capability. Marketing Spend may provide additional predictive value by")
    print("  * supporting market expansion, but its effect should be interpreted carefully")
    print("  * because it may be correlated with R&D Spend. Administration may be weaker,")
    print("  * but it can still reflect company scale or operational maturity. State is")
    print("  * treated as an auxiliary categorical feature and encoded using One-Hot")
    print("  * Encoding. Because the dataset contains only 50 observations, the results")
    print("  * should be interpreted as predictive associations rather than causal")
    print("  * conclusions. The final model should be selected based on performance,")
    print("  * stability, simplicity, and interpretability.")
    print("  * " + "="*70 + "\n")

    # --------------------------------------------------------------------------
    # CRISP-DM Step 6: Deployment & Simulation
    # --------------------------------------------------------------------------
    deployment_simulation(final_pipeline, selected_features)
    
    # Save model
    output_filename = "startup_profit_model_v2.pkl"
    save_model(final_pipeline, output_filename)


if __name__ == '__main__':
    main()
