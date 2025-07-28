# Predicting Glucose Levels with Bayesian Regression

This project explores Bayesian Ridge Regression* to predict average glucose levels in patients using clinical and demographic data. The model is compared with other regressors to evaluate its strengths and limitations in a healthcare setting.

#Objective
Use **Bayesian Regression** to predict `avg_glucose_level` from the Stroke Prediction Dataset and compare its performance with:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Why Bayesian Regression?
- Provides **probabilistic predictions** (not just point estimates)
- Handles **uncertainty** well
- Performs well with **small or noisy datasets**
- Useful for **clinical decision-making**

#Dataset
- Source: [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Target variable: `avg_glucose_level`
- Selected features: `age`, `bmi`, `hypertension`, `heart_disease`, `work_type`, etc.
- Feature engineering: interaction terms like `age * bmi`

# Model Comparison (Sample Results)

| Model              | MSE     | R² Score |
|-------------------|---------|----------|
| **Bayesian Ridge** | ~1933.9 | Low      |
| Linear Regression  | ~1936.1 | Low      |
| Decision Tree      | ~3441.8 | -0.73    |
| Random Forest      | ✅ Best | ✅ Best   |

- Random Forest had the best performance, but Bayesian Regression remains ideal when interpretability and uncertainty estimation are important.

# Visualizations
- Error distribution plots for all models
- Scatter plot: Actual vs Predicted (Bayesian-focused)
- Feature importance (Random Forest)

Files
- final_project.ipynb – Python code (training, evaluation, plotting)
- Bayesian_Regression_Report.pdf – Full written report


 How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python main.py
