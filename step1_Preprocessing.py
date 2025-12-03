import pandas as pd
import datetime

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
data.fillna(data.mean(numeric_only=True), inplace=True)

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

data.to_csv("cleaned_data.csv", index=False)

print("Step 1 completed successfully.")

#Edge for date 




    

    


