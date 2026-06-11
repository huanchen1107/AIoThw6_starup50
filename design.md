prompt_v2:
  title: "Kaggle 50 Startups CRISP-DM sklearn Project"
  version: "v2"
  role: >
    You are a professional data scientist, machine learning instructor,
    and multidisciplinary business analysis team.

  objective: >
    Write a complete Scikit-learn solution for the Kaggle 50 Startups dataset.
    The solution must strictly follow the CRISP-DM process and include clean
    Python code, expert-level feature analysis, model comparison, cross-validation,
    and careful business interpretation.

  dataset:
    name: "Kaggle 50 Startups"
    file_name: "50_Startups.csv"
    target_column: "Profit"
    features:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
      - "State"

  problem_type:
    learning_type: "Supervised Learning"
    task_type: "Regression"
    target: "Predict startup Profit"

  expert_panel:
    R&D_expert:
      focus:
        - "Product innovation"
        - "Technical capability"
        - "Product development"
        - "Long-term competitiveness"
      conclusion: >
        R&D Spend should be treated as the core predictive feature because it
        reflects the startup's innovation capability and product strength.

    marketing_expert:
      focus:
        - "Brand exposure"
        - "Customer acquisition"
        - "Market expansion"
        - "Advertising efficiency"
      conclusion: >
        Marketing Spend may help predict Profit because it can amplify the
        commercial impact of a strong product, but it should not be interpreted
        as an independent guarantee of higher profit.

    sales_expert:
      focus:
        - "Customer conversion"
        - "Revenue generation"
        - "Market demand"
        - "Sales performance"
      conclusion: >
        Marketing and R&D only create business value when they lead to actual
        customer conversion and revenue. Profit prediction should consider how
        spending may connect to business outcomes.

    regional_policy_expert:
      role_name: "Governor / Regional Policy Expert"
      focus:
        - "Regional business environment"
        - "Labor cost"
        - "Tax policy"
        - "Talent density"
        - "Startup ecosystem"
        - "Investment environment"
      conclusion: >
        State may capture regional conditions, but because the dataset has only
        50 observations and each State has limited samples, State should be
        treated only as an auxiliary feature.

    machine_learning_expert:
      focus:
        - "Feature importance"
        - "Overfitting risk"
        - "Multicollinearity"
        - "Cross-validation"
        - "Model stability"
        - "Interpretability"
      conclusion: >
        Because the dataset is small, model evaluation should not rely only on
        one train-test split. Use 5-fold cross-validation and select the final
        model based on predictive performance, stability, simplicity, and
        interpretability.

  expert_consensus:
    key_conclusion: >
      R&D Spend is the most likely dominant predictor of Profit. Marketing Spend
      may provide additional market expansion information. Administration and
      State should not be removed too early, but their value should be tested
      through model comparison. Because the dataset contains only 50 rows, all
      results should be interpreted as predictive associations rather than
      causal conclusions.

    feature_ranking:
      - rank: 1
        feature: "R&D Spend"
        consensus: "Core product and innovation factor"
        expected_importance: "High"
        recommendation: "Always keep"
      - rank: 2
        feature: "Marketing Spend"
        consensus: "Market expansion and customer acquisition factor"
        expected_importance: "Medium to high"
        recommendation: "Keep, but check correlation with R&D Spend"
      - rank: 3
        feature: "Administration"
        consensus: "Operating cost and company scale factor"
        expected_importance: "Low to medium"
        recommendation: "Keep first, then evaluate through model comparison"
      - rank: 4
        feature: "State"
        consensus: "Regional auxiliary factor"
        expected_importance: "Low to medium"
        recommendation: "Use One-Hot Encoding and avoid overinterpretation"

  crisp_dm_steps:
    step_1_business_understanding:
      requirements:
        - "Explain the business problem."
        - "Explain why predicting Profit is useful."
        - "Explain that startup resources are limited."
        - "Explain how the model may help founders, investors, analysts, and managers."
        - "Clearly state that this is a supervised regression problem."
        - "Include the multidisciplinary expert consensus."
      expert_notes:
        - "Do not overclaim causality."
        - "The dataset has only 50 rows."
        - "Use association and prediction language, not causal language."
        - "Emphasize that R&D Spend is expected to be the dominant predictor."

    step_2_data_understanding:
      requirements:
        - "Load the dataset using pandas."
        - "Show dataset shape."
        - "Show first five rows."
        - "Show data types."
        - "Check missing values."
        - "Check duplicate rows."
        - "Show descriptive statistics."
        - "Show State distribution."
        - "Show correlation matrix for numerical columns."
        - "Analyze Profit by State using groupby."
        - "Discuss the meaning of each feature using expert interpretation."
      code_checks:
        - "df.shape"
        - "df.head()"
        - "df.info()"
        - "df.describe()"
        - "df.isnull().sum()"
        - "df.duplicated().sum()"
        - "df['State'].value_counts()"
        - "df.corr(numeric_only=True)"
        - "df.groupby('State')['Profit'].agg(['count', 'mean', 'min', 'max', 'std'])"

    step_3_data_preparation:
      requirements:
        - "Separate X and y."
        - "Use Profit as target variable."
        - "Use R&D Spend, Administration, Marketing Spend, and State as features."
        - "Use OneHotEncoder for State."
        - "Do not use Label Encoding for State."
        - "Use ColumnTransformer."
        - "Use sklearn Pipeline."
        - "Use train_test_split with test_size=0.2 and random_state=42."
      preprocessing:
        numerical_features:
          - "R&D Spend"
          - "Administration"
          - "Marketing Spend"
        categorical_features:
          - "State"
        categorical_encoding:
          method: "OneHotEncoder"
          parameters:
            drop: "first"
            handle_unknown: "ignore"
        train_test_split:
          test_size: 0.2
          random_state: 42

    step_4_modeling:
      algorithm:
        primary_model: "LinearRegression"
        library: "scikit-learn"
        reason:
          - "The target variable Profit is continuous."
          - "The dataset is small."
          - "Linear Regression is interpretable."
          - "It is suitable for CRISP-DM teaching and reporting."
      model_experiments:
        model_1:
          name: "R&D Only"
          features:
            - "R&D Spend"
          purpose: "Check the predictive power of the core innovation feature."
          expert_question: "How much Profit can be explained by R&D Spend alone?"

        model_2:
          name: "R&D + Marketing"
          features:
            - "R&D Spend"
            - "Marketing Spend"
          purpose: "Check whether Marketing adds value beyond R&D."
          expert_question: "Does market expansion add predictive value after product innovation?"

        model_3:
          name: "Numerical Features"
          features:
            - "R&D Spend"
            - "Marketing Spend"
            - "Administration"
          purpose: "Check whether Administration improves prediction."
          expert_question: "Does operating cost or company scale add useful information?"

        model_4:
          name: "All Features"
          features:
            - "R&D Spend"
            - "Marketing Spend"
            - "Administration"
            - "State"
          purpose: "Check whether State improves prediction."
          expert_question: "Does regional information add useful predictive value?"

    step_5_evaluation:
      requirements:
        - "Evaluate each model with train-test split."
        - "Evaluate each model with 5-fold cross-validation."
        - "Print a comparison table."
        - "Select the final model based on CV R2 Mean, stability, simplicity, and interpretability."
        - "Discuss whether State adds meaningful value."
        - "Discuss whether R&D Spend dominates the prediction."
        - "Discuss whether Marketing Spend adds value beyond R&D."
        - "Discuss whether Administration is useful."
      metrics:
        - "R2 Score"
        - "MAE"
        - "RMSE"
        - "CV R2 Mean"
        - "CV R2 Std"
        - "CV RMSE Mean"
        - "CV RMSE Std"
      cross_validation:
        method: "KFold"
        parameters:
          n_splits: 5
          shuffle: true
          random_state: 42
      model_selection_rule:
        - "Prefer the model with strong CV R2 Mean."
        - "Prefer lower CV RMSE Mean."
        - "Check CV standard deviation to assess stability."
        - "If models are close, prefer the simpler and more interpretable model."
        - "Do not automatically assume the most complex model is best."
        - "Avoid overclaiming results because the dataset is small."

    step_6_deployment:
      requirements:
        - "Create a deployment simulation."
        - "Predict Profit for a new startup."
        - "Save the final model with joblib."
        - "Filename should be startup_profit_model_v2.pkl."
        - "Explain that this is a learning-project deployment, not full production deployment."
      sample_input:
        R&D Spend: 120000
        Administration: 130000
        Marketing Spend: 250000
        State: "New York"

  feature_analysis:
    R&D Spend:
      role: "Core innovation factor"
      expected_importance: "High"
      recommendation: "Always keep"
      interpretation: >
        R&D Spend is expected to be the strongest predictor because it reflects
        product development, innovation capability, technical competitiveness,
        and long-term growth potential.

    Marketing Spend:
      role: "Market expansion factor"
      expected_importance: "Medium to high"
      recommendation: "Keep, but check correlation with R&D Spend"
      interpretation: >
        Marketing Spend may help prediction by increasing market exposure and
        customer acquisition. However, it may also overlap with company size and
        R&D Spend. Avoid interpreting it as an independent causal factor.

    Administration:
      role: "Operating cost and company scale factor"
      expected_importance: "Low to medium"
      recommendation: "Keep first, evaluate later"
      interpretation: >
        Administration may be weaker because it does not directly create revenue,
        but it may still reflect company scale, management structure, and
        operational maturity.

    State:
      role: "Regional auxiliary factor"
      expected_importance: "Low to medium"
      recommendation: "Use One-Hot Encoding and avoid overinterpretation"
      interpretation: >
        State may reflect regional business environment, labor cost, tax policy,
        talent density, and investment ecosystem. However, because the dataset
        is small, State should be treated as an auxiliary variable only.

  machine_learning_warnings:
    small_sample_size:
      issue: "The dataset contains only 50 rows."
      action:
        - "Use 5-fold cross-validation."
        - "Check both mean and standard deviation of CV results."
        - "Avoid strong conclusions from a single train-test split."

    multicollinearity:
      issue: "R&D Spend and Marketing Spend may be correlated."
      action:
        - "Check correlation matrix."
        - "Avoid overinterpreting individual coefficients."
        - "Consider Ridge or Lasso in future versions."

    causality:
      issue: "This is observational data."
      action:
        - "Use predictive association language."
        - "Do not claim that one feature directly causes Profit to increase."

    model_complexity:
      issue: "More features do not always mean a better model."
      action:
        - "Compare feature sets."
        - "Prefer stable and interpretable models."
        - "Use the simplest model if performance is similar."

  code_requirements:
    language: "Python"
    libraries:
      - "pandas"
      - "numpy"
      - "joblib"
      - "scikit-learn"
    sklearn_modules:
      - "train_test_split"
      - "KFold"
      - "cross_val_score"
      - "ColumnTransformer"
      - "OneHotEncoder"
      - "Pipeline"
      - "LinearRegression"
      - "r2_score"
      - "mean_absolute_error"
      - "mean_squared_error"
    file_structure:
      output_file_name: "solve_50_startups_crispdm_v2.py"
      style:
        - "Use clear section comments."
        - "Separate helper functions."
        - "Use main() function."
        - "Use if __name__ == '__main__'."
        - "Print readable outputs."
        - "Handle missing file error."
        - "Check required columns before modeling."
        - "Include expert interpretation in printed output."

  required_functions:
    - name: "build_pipeline"
      purpose: "Build sklearn preprocessing and regression pipeline."
    - name: "evaluate_train_test"
      purpose: "Evaluate model with R2, MAE, and RMSE on test set."
    - name: "evaluate_cross_validation"
      purpose: "Evaluate model with 5-fold CV using R2 and RMSE."
    - name: "data_understanding"
      purpose: "Print dataset overview and exploratory checks."
    - name: "run_model_experiments"
      purpose: "Train and evaluate four feature-set models."
    - name: "select_final_model"
      purpose: "Select best model based on CV R2 Mean, stability, and interpretability."
    - name: "deployment_simulation"
      purpose: "Predict Profit for a new startup."
    - name: "save_model"
      purpose: "Save final pipeline using joblib."
    - name: "main"
      purpose: "Run the complete CRISP-DM workflow."

  output_requirements:
    must_include:
      - "Complete Python code."
      - "CRISP-DM comments."
      - "Business understanding explanation."
      - "Multidisciplinary expert consensus."
      - "Machine learning expert warnings."
      - "Data understanding outputs."
      - "Four model experiments."
      - "Evaluation table."
      - "Expert interpretation."
      - "Deployment simulation."
      - "Saved model."
    output_format:
      - "Return the final Python code in one code block."
      - "Do not skip any CRISP-DM step."
      - "Make the code runnable as a single script."

  final_interpretation_template: >
    The model comparison shows that R&D Spend is expected to be the most important
    predictor of Profit because it reflects product innovation and technical
    capability. Marketing Spend may provide additional predictive value by
    supporting market expansion, but its effect should be interpreted carefully
    because it may be correlated with R&D Spend. Administration may be weaker,
    but it can still reflect company scale or operational maturity. State is
    treated as an auxiliary categorical feature and encoded using One-Hot
    Encoding. Because the dataset contains only 50 observations, the results
    should be interpreted as predictive associations rather than causal
    conclusions. The final model should be selected based on performance,
    stability, simplicity, and interpretability.

  constraints:
    - "Do not use Label Encoding for State."
    - "Do not claim causality."
    - "Do not remove Administration too early."
    - "Do not rely only on one train-test split."
    - "Use cross-validation because the dataset is small."
    - "Do not automatically choose the most complex model."
    - "Check whether State actually improves prediction."
    - "Keep the solution beginner-friendly but professionally structured."