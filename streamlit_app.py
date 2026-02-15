# --- Code Cell ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Code Cell ---
df = pd.read_csv("Data/churn.csv")

# --- Code Cell ---
df.head()

# --- Code Cell ---
df.tail()

# --- Code Cell ---
print(df.info())

# --- Code Cell ---
print(df["Churn"].value_counts())

# --- Code Cell ---
df.describe()

# --- Code Cell ---
# Drop ID column
df = df.drop("customerID", axis=1)


# --- Code Cell ---
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# --- Code Cell ---
# Fill missing values (FIXED VERSION)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# --- Code Cell ---
# Convert target column
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# --- Code Cell ---
print(df)

# --- Code Cell ---
#Data Visualization(Target Distribution)
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()


# --- Code Cell ---
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# --- Code Cell ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])


# --- Code Cell ---
from sklearn.model_selection import train_test_split

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- Code Cell ---
X = df.drop("Churn", axis=1)
y = df["Churn"]

# --- Code Cell ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Code Cell ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# --- Code Cell ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


# --- Code Cell ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))


# --- Code Cell ---
import joblib

for name, model in models.items():
    joblib.dump(model, f"Models/{name}.pkl")


# --- Code Cell ---


# --- Code Cell ---


