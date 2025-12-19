import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data with cluster labels
data = pd.read_csv("clustered_data.csv")

#Input
X = data[[
     'annual_income',
    'total_spend',
    'total_purchases',
    'family_size',
    'age',
    'num_discount_purchases',
    'days_since_last_purchase',
    'web_visits_last_month'
]]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
#Output
y = data["cluster"]

#80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

#scaling (for KNN - decision tree doenst need sclaing, but it wont hurt.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)

#fit data into area
knn.fit(X_train_scaled, y_train)

# Predict
y_pred_knn = knn.predict(X_test_scaled)

#Evaluation
print("KNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn,  pos_label=1))  #cluster 1 label is considered “positive.”
print("Recall:", recall_score(y_test, y_pred_knn,  pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_knn,  pos_label=1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

# Predict
y_pred_dt = dt.predict(X_test)

#Evaluation/
print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt,  pos_label=1))
print("Recall:", recall_score(y_test, y_pred_dt, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_dt, pos_label=1 ))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

#Random Forest : MANY decision trees, each tree gets result,  majority voting on results -> stable outcome , prevents overfitting

# ============================================
# ✦ BONUS MODEL — RANDOM FOREST
# ============================================

# Random Forest does NOT need scaling, but you can use scaled or unscaled. 
# We use unscaled (the original X_train, X_test) for best performance.
rf = RandomForestClassifier(
    n_estimators=200,        # more trees = better accuracy (200 trees)
    max_depth=None,         # let trees fully grow
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results (BONUS):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf,  pos_label=1))
print("Recall:", recall_score(y_test, y_pred_rf,  pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_rf,  pos_label=1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#NEW
# Predict cluster for the new user using KNN, Decision Tree and Random Forest models:   
#new_user = X.tail(1)                  
new_user = data[[
    'annual_income',
    'total_spend',
    'total_purchases',
    'family_size',
    'age',
    'num_discount_purchases',
    'days_since_last_purchase',
    'web_visits_last_month'
]].tail(1)

# Apply SAME imputer and scaler
new_user_imputed = imputer.transform(new_user)
new_user_scaled = scaler.transform(new_user_imputed)
# For Decision Tree and Random Forest, use unscaled
new_user_array = new_user_imputed  # define it properly

print("\nModel Prediction for NEW USER:")
print("KNN:", knn.predict(new_user_scaled)[0])
print("Decision Tree:", dt.predict(new_user_array)[0])
print("Random Forest:", rf.predict(new_user_array)[0])
