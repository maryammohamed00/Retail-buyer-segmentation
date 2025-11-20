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

