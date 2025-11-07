import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

#  Load Dataset
data = pd.read_csv('house_prices_data.csv')
print("\tData Loaded Successfully!")
print("Total Rows:", len(data))
print(data.head())

# lowercase
data['City'] = data['City'].str.lower()
data['Price'] = data['Price'].fillna(data['Price'].mean())

#  Split Features & Target
X = data[['City', 'Area', 'Bedrooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data: {X_train.shape}, Test Data: {X_test.shape}")

# Preprocessing Setup
numeric_features = ['Area', 'Bedrooms']
categorical_features = ['City']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'  #  this prevents the crash
    ))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

#  Model Pipeline
model = Pipeline([
    ('preprocessing', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=150, random_state=42))
])

#  Train Model
print("\n\tTraining Model...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



#  performance

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("\n\tModel Performance")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

#  Save Model
joblib.dump(model, "House_model.joblib")
print("\nModel Saved Successfully as 'House_model.joblib'.")

#  User Input
city_name = pd.DataFrame({
    'City Name': ["Delhi", 'Mumbai', 'Lucknow', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Pune', 'Jaipur', 'Ahmedabad']
})
print("\nCity Name:\n", city_name)

City = input("Enter City name According to Given City: ").strip().lower()
Area = input("Enter Total area: ")
Bedrooms = input("Enter Bedrooms : ")

#  Prediction
if Area.isnumeric() and Bedrooms.isnumeric():
    Area = float(Area)
    Bedrooms = int(Bedrooms)
    
    if City in city_name['City Name'].str.lower().values:
        new_data = pd.DataFrame([[City, Area, Bedrooms]], columns=['City', 'Area', 'Bedrooms'])
        preds = model.predict(new_data)[0]
        print("\n\tPredicted Price:", round(preds, 2), "Lakh")
    else:
        print("This City is not available right now.")
else:
    print("Please enter valid numeric values for Area and Bedrooms.")#  Evaluate Model
