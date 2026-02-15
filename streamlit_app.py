
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("Telco Customer Churn Prediction")
st.write("Classification using 6 Machine Learning Models")

os.makedirs("Models", exist_ok=True)

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # ---------------- Split ----------------
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    model_name = st.selectbox(
        "Select Model",
        ["Logistic", "DecisionTree", "KNN", "NaiveBayes", "RandomForest", "XGBoost"]
    )

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    if st.button("Train Selected Model"):

        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        
        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))
        st.write("AUC:", roc_auc_score(y_test, y_pred))
        st.write("MCC:", matthews_corrcoef(y_test, y_pred))

       
        joblib.dump(model, f"Models/{model_name}.pkl")

        
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
