import pandas as pd
#NEW
import numpy as np 
import datetime
# Add these imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Replace 'your_file.csv' with the path to your training data file
data = pd.read_csv('data.csv')

# Show the first 5 rows
print(data.head())

# Show general info about the dataset (columns, data types, missing values)
print(data.info())

# Show basic statistics for numeric columns
print(data.describe())

# Check missing values per column
print(data.isnull().sum())

#Replace missing values with the mean of its column if numeric. 
#data.fillna(data.mean(numeric_only=True), inplace=True)


#NEW
#Replace missing values with the mode of its column if numeric or text.
#data.fillna(data.mode().iloc[0], inplace=True)

#fix 1 - for handling outliers and missing values - K
# 1️⃣ Model-based imputation for numeric columns
numeric_cols = data.select_dtypes(include=["number"]).columns

# Iterative Imputer (MICE)
imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
imputer_cols = numeric_cols.copy()

categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])


print("Duplicated rows:",data.duplicated().sum())
data = data.drop_duplicates()

#convert non-numeric data types to numeric(boolean). 
data = pd.get_dummies(data, columns=["education_level", "marital_status"])

"""Feature Engineering"""

# 1.2 Customer tenure (how long they've been a customer)
# Choose a consistent reference date (any fixed date in the future)

today = pd.Timestamp(datetime.date.today())
data["age"] = today.year - data["birth_year"]


#Features: 
spend_cols = [
    'spend_wine', 'spend_meat', 'spend_fruits',
    'spend_fish', 'spend_sweets', 'spend_gold'
]
data["total_spend"] = data[spend_cols].sum(axis=1)  #axis 0; Vertical Sum - axis 1; Horizontal sum

purchase_cols = [
    'num_web_purchases', 'num_store_purchases', 'num_catalog_purchases'
]
data["total_purchases"] = data[purchase_cols].sum(axis=1)

# 4. Family Size (children + teenagers)
data["family_size"] = data["num_children"] + data["num_teenagers"]  

# 5. Drop raw date column (not useful for modeling anymore)
data.drop(columns=["signup_date"], inplace=True)

#Outlier Handling:
def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
       #Clipping -> Decreasing upper outlier to our max val. and increasing lower outlier to our min val.


#Loop to apply outlier handling on all necessary columns.
for col in ['annual_income', 'spend_wine', 'spend_meat',
            'spend_fruits','spend_fish', 'spend_sweets', 'spend_gold', 
            'num_web_purchases', 'num_store_purchases', 
            'num_catalog_purchases','num_discount_purchases',
            'days_since_last_purchase','web_visits_last_month',
            'total_spend', 'total_purchases', 'age','family_size']:
    if col in data.columns:
        cap_outliers_iqr(data, col)

#Edge for date 


# ============================
#   ADDED: USER INPUT SECTION
# ============================

print("\n--- Add NEW User Data ---")
print("Enter customer info (you may leave blank or enter anything):")

user_input = {
    "annual_income": input("Annual income: "),
    "total_spend": input("Total spend: "),
    "total_purchases": input("Total purchases: "),
    "family_size": input("Family size: "),
    "age": input("Age: "),
    "num_discount_purchases": input("Discount purchases: "),
    "days_since_last_purchase": input("Days since last purchase: "),
    "web_visits_last_month": input("Web visits last month: ")
}

user_row = pd.DataFrame([user_input])

# ----------------------------
# AUTO CLEANING (STEP 1)
# ----------------------------

user_numeric_cols = [
    "annual_income", "total_spend", "total_purchases", "family_size",
    "age", "num_discount_purchases", "days_since_last_purchase",
    "web_visits_last_month"
]

for col in user_numeric_cols:
    # Convert invalid input → NaN
    user_row[col] = pd.to_numeric(user_row[col], errors="coerce")

    # Ensure no negatives
    user_row[col] = user_row[col].clip(lower=0)

# ----------------------------
# ALIGN FEATURES (STEP 2)  ✅ KEY FIX
# ----------------------------

# Get ALL numeric columns used during training
trained_numeric_cols = data.select_dtypes(include="number").columns

# Add missing numeric columns expected by the imputer
for col in trained_numeric_cols:
    if col not in user_row.columns:
        user_row[col] = 0

# Reorder columns to match training data
user_row = user_row[trained_numeric_cols]

# ----------------------------
# IMPUTATION (STEP 3)  ✅ APPLY ONCE
# ----------------------------

user_row[imputer_cols] = imputer.transform(
    user_row[imputer_cols].fillna(0)
)

# ----------------------------
# APPEND TO DATASET
# ----------------------------

data = pd.concat([data, user_row], ignore_index=True)

# ----------------------------
# OUTLIER HANDLING
# ----------------------------

for col in ['annual_income', 'spend_wine', 'spend_meat',
            'spend_fruits','spend_fish', 'spend_sweets', 'spend_gold', 
            'num_web_purchases', 'num_store_purchases', 
            'num_catalog_purchases','num_discount_purchases',
            'days_since_last_purchase','web_visits_last_month',
            'total_spend', 'total_purchases', 'age','family_size']:
    if col in data.columns:
        cap_outliers_iqr(data, col)

# ----------------------------
# SAVE OUTPUT
# ----------------------------

data.to_csv("cleaned_data.csv", index=False)

print("\nUser record cleaned and added successfully!")
print(user_row)

print("\n✓ Step 1 Finalized (with user input).")
