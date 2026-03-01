# 💼 Adult Income Classification using Gaussian Naive Bayes

This project applies the Gaussian Naive Bayes algorithm to predict whether a person earns more than 50K per year using the Adult Income dataset. The workflow includes data exploration, preprocessing, encoding, scaling, model training, evaluation, ROC analysis, and cross-validation.

## 📌 Project Objective
The goal is to build a classification model that predicts income category:
- <=50K
- >50K

The dataset contains demographic and employment-related attributes such as age, education, occupation, marital status, and working hours.

## 📊 Dataset Overview
The dataset consists of 32,561 observations and 15 features:
- 6 numerical features (age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week)
- 9 categorical features (workclass, education, marital_status, occupation, relationship, race, sex, native_country, income)

The target variable is `income`.

## 🔍 Data Exploration & Understanding
The dataset was explored to:
- Identify feature types (categorical vs numerical)
- Examine value distributions
- Detect missing values
- Understand class distribution

The income distribution shows class imbalance:
- Majority class: <=50K
- Minority class: >50K

## 🧹 Handling Missing Values
Missing values were identified in:
- workclass
- occupation
- native_country

Missing values were replaced using the mode of each feature (most frequent category) to preserve dataset size and maintain distribution consistency.

## 🏷️ Feature Encoding
Since machine learning models require numerical input:
- All categorical variables were encoded using One-Hot Encoding.
- This avoids imposing ordinal relationships between categories.

After encoding, the dataset expanded to 105 features.

## ⚖️ Feature Scaling
RobustScaler was applied to:
- Reduce the influence of outliers
- Improve numerical stability for Gaussian Naive Bayes

## 🤖 Model Selection: Gaussian Naive Bayes
Gaussian Naive Bayes was chosen because:
- It performs well with numerical data
- It assumes features follow a Gaussian distribution
- It is computationally efficient
- It works well for high-dimensional datasets

The model estimates:
- Prior probabilities for each class
- Conditional probabilities for each feature

## 📈 Model Performance

### Accuracy
- Training Accuracy ≈ 80.67%
- Testing Accuracy ≈ 80.83%

### Null Accuracy
Null accuracy (predicting majority class only) ≈ 75.82%

Since model accuracy is significantly higher than null accuracy, the classifier provides meaningful predictive power.

## 📊 Confusion Matrix Analysis
Confusion Matrix Results:

- True Positives: 5999
- True Negatives: 1897
- False Positives: 1408
- False Negatives: 465

The model shows strong recall for the <=50K class but moderate precision for the >50K class.

## 📑 Classification Report
- Overall Accuracy ≈ 81%
- Balanced performance across classes
- Better recall for minority class compared to naive baseline

## 📉 ROC Curve & AUC
ROC curve was plotted to evaluate classification performance across thresholds.

- ROC AUC ≈ 0.8941
- Cross-Validated ROC AUC ≈ 0.8938

This indicates strong separability between income classes.

## 🔁 Cross-Validation
10-Fold Cross-Validation Accuracy ≈ 80.63%

This confirms model stability and generalization ability.

## 📊 Probability Distribution Analysis
Predicted probabilities were visualized using histograms to understand how confidently the model predicts income >50K.

## ⚠️ Limitations
- Naive Bayes assumes feature independence, which may not hold in real-world socioeconomic data.
- Accuracy alone does not capture class imbalance fully.
- More advanced models (e.g., Random Forest, Gradient Boosting) may achieve higher performance.

## 🛠️ Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Category Encoders.

## 🎯 Skills Demonstrated
Data Cleaning, Handling Missing Values, Categorical Encoding, Feature Scaling, Probabilistic Modeling, Confusion Matrix Analysis, ROC-AUC Evaluation, Cross-Validation, and Performance Interpretation.
