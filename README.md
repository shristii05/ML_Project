# ML_Project
# ML Model Comparison Project

This repository demonstrates the implementation and comparison of several fundamental machine learning algorithms on different real-world datasets. The goal is to evaluate their performance and understand which models are best suited for particular types of data and problems.

---

## Project Overview

Machine learning models behave differently depending on the type of data, its features, and the problem (classification or regression). In this project, we:

- Explore and preprocess multiple datasets.
- Implement key ML models using `scikit-learn`.
- Evaluate models using appropriate metrics.
- Compare results to determine which model performs best in each scenario.

---

## Models Implemented

1. **Naive Bayes** - Probabilistic classifier based on Bayes’ theorem.
2. **Random Forest** - Ensemble of decision trees using bagging.
3. **Decision Tree** - Tree-based flowchart for decision making.
4. **Linear Regression** - Predicts continuous values based on linear relationships.
5. **Logistic Regression** - Predicts binary/class labels using a sigmoid function.
6. **Ensemble Learning** - Combines multiple models to improve predictions (Voting, Bagging, Boosting).

---

## Datasets Used

| Model                | Dataset Description | Source |
|---------------------|----------------------|--------|
| **Naive Bayes**      | UCI Census Income Dataset — predict whether income >50K. | [adult.data.csv](https://raw.githubusercontent.com/anshu7vyas/naive-bayesian-census-income/refs/heads/master/Data/adult.data.csv) |
| **Random Forest**    | Drug classification based on age, sex, BP, cholesterol, and Na levels. | [drug200.csv](https://raw.githubusercontent.com/ajaychouhan-nitbhopal/Drug-Classification-with-Random-forest-Classifier-on-Drug200-dataset/refs/heads/main/(2)%20drug200.csv) |
| **Ensemble Learning**| Loan approval prediction from financial and demographic info. | [train_clean_data.csv](https://raw.githubusercontent.com/madhur02/ensemble-learning-loan-dataset/refs/heads/master/train_clean_data.csv) |
| **Logistic Regression** | Heart disease prediction from clinical parameters. | [heart.csv](https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/refs/heads/master/heart.csv) |
| **Linear Regression** | Predict advertising sales based on spending across channels. | [Advertising.csv](https://raw.githubusercontent.com/erkansirin78/datasets/master/Advertising.csv) |
| **Decision Tree**    | Predict diabetes from health metrics like glucose and BMI. | [diabetes_dataset.csv](https://raw.githubusercontent.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/refs/heads/master/diabetes_dataset.csv) |

---

## Evaluation Metrics

**Classification Models** (Naive Bayes, Random Forest, Decision Tree, Logistic Regression, Ensemble):

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC (if applicable)

**Regression Model** (Linear Regression):

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

---

