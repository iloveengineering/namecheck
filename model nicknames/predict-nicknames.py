import pandas as pd
import joblib

# Load the trained model
model = joblib.load('train_model_nickname.pkl')  # Replace with the actual path to your trained model

# Input the name and nickname for prediction
name = input("Enter the name: ")
nickname = input("Enter the nickname: ")

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'First Name': [name.split()[0]],
    'Middle Name': [name.split()[1] if len(name.split()) > 1 else ''],
    'Surname': [name.split()[2] if len(name.split()) > 2 else ''],
    'Nicknames': [nickname]
})

# Use the trained model for prediction
probability = model.predict(input_data)

# Display the predicted probability
print(f"The predicted probability is: {probability[0]}")
