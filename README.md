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






