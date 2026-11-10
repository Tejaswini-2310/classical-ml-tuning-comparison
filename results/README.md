# Default vs Tuned Classical Machine Learning Models

## Project Overview

This project compares default and hyperparameter-tuned classical machine learning algorithms across small tabular datasets.

The objective is to investigate whether hyperparameter tuning consistently improves model performance.

## Algorithms Used

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

## Datasets

- Breast Cancer
- Wine
- Pima Diabetes
- Heart Statlog
- German Credit (Credit-G)

Some datasets are downloaded from OpenML and require an internet connection.

## Methodology

The experiment uses nested cross-validation:

- Outer cross-validation: 5 folds
- Inner cross-validation: 3 folds
- Repetitions: 5
- Hyperparameter optimization: RandomizedSearchCV
- Search candidates: 15 per inner search

The same outer splits are used to compare default and tuned configurations.

## Evaluation Metrics

- Accuracy
- Macro-F1 Score
- ROC-AUC

## Technologies

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- SciPy

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Experiment

```bash
python tuning_vs_default_experiment.py
```

## Results

The experiment generates:

- Raw cross-validation scores
- Mean and standard deviation summary
- Paired F1-score improvements
- Model ranking comparisons
- Default vs tuned F1 graph
- Ranking heatmap
- F1 improvement boxplot

All generated results are saved in the `results/` directory.

## Key Findings

Hyperparameter tuning does not consistently improve every model across all datasets. Its effectiveness depends on the dataset and algorithm.

The experiment highlights the importance of empirical model evaluation and fair comparisons using nested cross-validation.

## Author

Tejaswini Reddy

GitHub: https://github.com/Tejaswini-2310