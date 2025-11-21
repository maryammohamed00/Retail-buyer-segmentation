import pandas as pd

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

#Function: Handle Outliers by Computecomputing outliers and clipping them. 
def remove_outliers_iqr(data, column):
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1 #Inter-Quartile Range

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

#Loop to apply outlier handling on all necessary columns.
for col in ['annual_income', 'spend_wine', 'spend_meat',
            'spend_fruits','spend_fish', 'spend_sweets', 'spend_gold', 
            'num_web_purchases', 'num_store_purchases', 
            'num_catalog_purchases','num_discount_purchases',
            'days_since_last_purchase','web_visits_last_month']:
    
    data = remove_outliers_iqr(data, col)

data.to_csv("cleaned_data.csv", index=False)

print("Step 1 completed successfully.")

#hi #hello :)
    # Apply outlier on all numerical values?
    # and what about age, cildren, teenage count?
    
    


