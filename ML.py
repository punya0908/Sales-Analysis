# ============================================
# UNIT I: DATA LOADING & PREPROCESSING
# ============================================

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_excel(r"C:\Users\PUNYA\OneDrive - Lovely Professional University\Desktop\Dispatch order Office 2017.xlsx", sheet_name="Summary Nov 19")

# Identify product quantity columns
product_cols = df.columns[9:-1]

# Convert product columns to numeric, handling errors
for col in product_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
df[product_cols] = df[product_cols].fillna(0)

# Convert date if column exists
if 'Outward Date' in df.columns:
    df['Outward Date'] = pd.to_datetime(
        df['Outward Date'], format='%d-%b', errors='coerce'
    )
    df['Month'] = df['Outward Date'].dt.month
else:
    df['Month'] = 0

# Recalculate total
df['Total_Calc'] = df[product_cols].sum(axis=1)

# Encode categorical variables if they exist
cat_cols = [col for col in ['city', 'STT'] if col in df.columns]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Drop non-useful columns
df.drop(columns=['Party Name', 'B.No', 'Chq', 'GST', 'Outward Date'], errors='ignore', inplace=True)

print("UNIT I COMPLETE: Data Preprocessing Done")

# ============================================
# UNIT II: SUPERVISED LEARNING – REGRESSION
# ============================================

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Use Total_Calc if Total doesn't exist
if 'Total' not in df.columns:
    df['Total'] = df['Total_Calc']

X = df.drop(columns=['Total', 'Total_Calc'], errors='ignore')
y = df['Total']

# Select only numeric columns for X
X = X.select_dtypes(include=[np.number])

# Fill any remaining NaN values
X = X.fillna(0)
y = y.fillna(0)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Models
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
svr = SVR()

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('lr', LinearRegression())
])

models = {
    "Linear Regression": lr,
    "Polynomial Regression": poly_model,
    "Decision Tree": dt,
    "KNN": knn,
    "SVR": svr
}

def evaluate_regression(name, y_true, y_pred):
    print(f"\n{name}")
    print("MAE :", round(mean_absolute_error(y_true, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("R2  :", round(r2_score(y_true, y_pred), 4))

reg_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    evaluate_regression(name, y_test, preds)
    reg_results[name] = r2_score(y_test, preds)

print("\nBEST REGRESSION MODEL:", max(reg_results, key=reg_results.get))

# ============================================
# UNIT III: SUPERVISED LEARNING – CLASSIFICATION
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Create classification target
def order_class(total):
    if total <= 5:
        return 0   # Small
    elif total <= 15:
        return 1   # Medium
    else:
        return 2   # Large

df['Order_Class'] = df['Total'].apply(order_class)

X_cls = df.drop(columns=['Order_Class', 'Total', 'Total_Calc', 'Cluster'], errors='ignore')
# Select only numeric columns
X_cls = X_cls.select_dtypes(include=[np.number])
# Fill any remaining NaN values
X_cls = X_cls.fillna(0)
y_cls = df['Order_Class']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

scaler_cls = StandardScaler()
Xc_train = scaler_cls.fit_transform(Xc_train)
Xc_test = scaler_cls.transform(Xc_test)

cls_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

cls_results = {}

for name, model in cls_models.items():
    model.fit(Xc_train, yc_train)
    preds = model.predict(Xc_test)
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(yc_test, preds), 4))
    print(classification_report(yc_test, preds))
    cls_results[name] = accuracy_score(yc_test, preds)

print("\nBEST CLASSIFICATION MODEL:", max(cls_results, key=cls_results.get))

# ============================================
# UNIT IV: UNSUPERVISED LEARNING
# ============================================

from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans

# Get existing product columns only
existing_product_cols = [col for col in product_cols if col in df.columns]
# Ensure product columns are numeric
product_data = df[existing_product_cols].select_dtypes(include=[np.number])
product_data = product_data.fillna(0)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(product_data)

# Hierarchical Clustering
Z = linkage(product_data, method='ward')

print("UNIT IV COMPLETE: Clustering Done")

# ============================================
# UNIT V: PCA & NEURAL NETWORKS
# ============================================

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(product_data)

# Neural Network
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 30),
    max_iter=1000,
    random_state=42
)

mlp.fit(Xc_train, yc_train)
mlp_preds = mlp.predict(Xc_test)

print("\nMLP Accuracy:", round(accuracy_score(yc_test, mlp_preds), 4))

# ============================================
# UNIT VI: MODEL PERFORMANCE & ENSEMBLES
# ============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Cross Validation
cv_score = cross_val_score(
    DecisionTreeClassifier(random_state=42),
    Xc_train, yc_train, cv=5
).mean()

print("\nCross Validation Accuracy:", round(cv_score, 4))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(Xc_train, yc_train)
rf_preds = rf.predict(Xc_test)

print("Random Forest Accuracy:", round(accuracy_score(yc_test, rf_preds), 4))

print("\nPROJECT EXECUTION COMPLETE")
