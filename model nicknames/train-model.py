import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load your training data from CSV
training_data = pd.read_csv('your_dataset.csv')

# Extract features and target variable
features = ['First Name', 'Middle Name', 'Surname', 'Nicknames']
target = 'Probability'

# Create a copy to avoid the SettingWithCopyWarning
X_train = training_data[features].copy()
y_train = training_data[target]

# Separate numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['number']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Create transformers for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Replace with your preferred imputation strategy
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Train a model (replace RandomForestRegressor with your chosen algorithm)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Fit the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'train_model_nickname.pkl')
