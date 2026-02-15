Telco Customer Churn Classification Project

->Problem Statement
The goal of this project is to build multiple machine learning
classification models to predict customer churn in a telecom company.
Customer churn refers to customers who stop using a company's service.
Identifying such customers helps organizations take preventive actions
and improve the retention policy

->Dataset Description
This project uses the Telco Customer Churn dataset which contains
information about customers, including demographic details, account
information, and service usage patterns.

->Total Records: 7000+
->Features: 20+
->Target Variable: Churn (Yes/No)
->Type: Binary Classification Problem

Key features include: - Gender - SeniorCitizen - Tenure -
MonthlyCharges - TotalCharges - Contract Type - Internet Service -
Payment Method

->Data Preprocessing
The following steps were performed to clean and prepare the data: -
Removed the customerID column as it does not contribute to prediction -
Converted TotalCharges to numeric format - Filled missing values using
median - Encoded categorical variables using label encoding / one-hot
encoding - Scaled numerical features using StandardScaler - Split data
into training and testing sets

->Models Used
The following six classification models were implemented:

1.Logistic Regression
2.Decision Tree Classifier
3.K-Nearest Neighbors (KNN)
4.Naive Bayes (GaussianNB)
5.Random Forest (Ensemble)
6.XGBoost (Ensemble)

->Evaluation Metrics

Each model was evaluated using the following metrics: - Accuracy - AUC
Score - Precision - Recall - F1 Score - Matthews Correlation Coefficient
(MCC)

->Model Comparison Table

  -----------------------------------------------------------------------------
  ML Model Name          Accuracy   AUC   Precision   Recall   F1 Score   MCC
  ---------------------- ---------- ----- ----------- -------- ---------- -----
  Logistic Regression     0.799    0.718    0.642      0.548    0.5916    0.4621

  Decision Tree           0.726    0.652    0.4855     0.4946   0.4900    0.3034

  KNN                     0.7423   0.665    0.515      0.5026   0.5087    0.3342

  Naive Bayes             0.7466   0.7421   0.5160     0.7326   0.6055    0.4413

  Random Forest           0.7870   0.6928   0.625      0.4919   0.5508    0.4191

  XGBoost                 0.7785   0.6947   0.5956     0.5160   0.5530    0.4086
  -----------------------------------------------------------------------------



->Observations

  -----------------------------------------------------------------------
  ML Model Name             Observation about Model Performance
  ------------------------- ---------------------------------------------
  Logistic Regression       Works well as a baseline model and gives stable performance
  Decision Tree             Can overfit if depth is high but captures feature relationships
  KNN                       Sensitive to scaling and works better with optimal k
  Naive Bayes               Fast and simple but assumes feature independence
  Random Forest             High accuracy and handles overfitting better
  XGBoost                   Usually gives best performance with tuned parameters
  -----------------------------------------------------------------------

->Project Structure

project-folder
│-- app.py
│-- requirements.txt
│-- README.md
│-- model

