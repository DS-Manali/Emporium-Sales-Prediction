# Emporium-Sales-Prediction
Objective: Build a model which predicts sales based on various factors of emporium and products.

This helps the sales team to understand the best and worst-selling products and products to promote

Finding the best-suited model 
pd.crosstab(dfo[i], dfo['Install_Week_Number'], margins=True, margins_name="Total")
    dfo.groupby(['i']).agg({'table1':'sum', 'AppsFlyer ID': 'count'}).reset_index()




import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
df = pd.read_csv("your_data.csv")  # Replace with actual file

# Step 1: Identify Correct Data Types
def infer_dtype(series):
    try:
        # Try converting to numeric
        pd.to_numeric(series)
        return 'numeric'
    except ValueError:
        pass

    try:
        # Try converting to boolean
        if set(series.dropna().unique()) <= {0, 1, '0', '1', True, False}:
            return 'bool'
    except ValueError:
        pass

    try:
        # Try converting to datetime
        pd.to_datetime(series, errors='coerce')
        return 'date'
    except ValueError:
        pass

    return 'object'

# Apply Data Type Inference
dtype_dict = {col: infer_dtype(df[col]) for col in df.columns}

# Convert Columns Based on Detected Type
for col, dtype in dtype_dict.items():
    if dtype == 'numeric':
        df[col] = pd.to_numeric(df[col])
    elif dtype == 'bool':
        df[col] = df[col].astype(int)  # Convert True/False to 1/0
    elif dtype == 'date':
        df[col] = pd.to_datetime(df[col], errors='coerce')
    # Object columns remain unchanged

# Step 2: Handle Missing Values
for col in df.columns:
    if dtype_dict[col] == 'numeric':
        df[col].fillna(df[col].median(), inplace=True)
    elif dtype_dict[col] == 'bool':
        df[col].fillna(df[col].mode()[0], inplace=True)
    elif dtype_dict[col] == 'date':
        df[col].fillna(df[col].min(), inplace=True)  # Fill missing dates with earliest date
    elif dtype_dict[col] == 'object':
        df[col].fillna('Unknown', inplace=True)  # Fill categorical missing values with 'Unknown'

# Encode Categorical Columns
obj_cols = [col for col, dtype in dtype_dict.items() if dtype == 'object']
for col in obj_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Step 3: Define Features and Target
X = df.drop(columns=['target'])  # Replace 'target' with your actual target column
y = df['target']

# Step 4: Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', importance_type='gain')
xgb_model.fit(X_train, y_train)

# Get Feature Importance
feature_importance = xgb_model.get_booster().get_score(importance_type="gain")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Step 5: Get Top 20 Features
top_20_features = pd.DataFrame(sorted_features[:20], columns=['Feature', 'Importance'])
print(top_20_features)


















import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# Information Value Function
def calculate_iv(df, feature, target):
    """Calculate Information Value (IV) for a feature"""
    eps = 1e-10  # Small value to avoid division by zero
    df['bin'] = pd.qcut(df[feature].rank(method="first"), q=10, duplicates='drop')
    grouped = df.groupby('bin')[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['bad_dist'] = grouped['bad'] / (grouped['bad'].sum() + eps)
    grouped['good_dist'] = grouped['good'] / (grouped['good'].sum() + eps)
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'] + eps)
    grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    return grouped['iv'].sum()

# Load Data
df = pd.read_csv("your_data.csv")  # Replace with your actual file

# Identify Data Types
num_cols = df.select_dtypes(include=['number']).columns.tolist()
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
bool_cols = df.select_dtypes(include=['bool']).columns.tolist()

# Convert Boolean to Integer
df[bool_cols] = df[bool_cols].astype(int)

# Encode Categorical Columns
label_encoders = defaultdict(LabelEncoder)
for col in obj_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Define Features and Target
X = df.drop(columns=['target'])  # Replace 'target' with actual target column
y = df['target']

# Calculate IV for Each Feature
iv_dict = {col: calculate_iv(df[[col, 'target']], col, 'target') for col in num_cols + obj_cols + bool_cols}
iv_sorted = sorted(iv_dict.items(), key=lambda x: x[1], reverse=True)

# Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', importance_type='gain')
xgb_model.fit(X_train, y_train)

# Get Feature Importance from XGBoost
xgb_importance = xgb_model.get_booster().get_score(importance_type="gain")
xgb_importance_sorted = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)

# Merge IV and XGBoost Importance
feature_importance_df = pd.DataFrame(iv_sorted, columns=['feature', 'IV'])
feature_importance_df['XGB_Importance'] = feature_importance_df['feature'].map(dict(xgb_importance_sorted))
feature_importance_df['XGB_Importance'].fillna(0, inplace=True)

# Rank Features Based on IV and XGBoost Importance
feature_importance_df['Final_Score'] = feature_importance_df['IV'] + feature_importance_df['XGB_Importance']
feature_importance_df = feature_importance_df.sort_values(by='Final_Score', ascending=False)

# Select Top 20 Features
top_20_features = feature_importance_df.head(20)
print(top_20_features)



