import pandas as pd
from faker import Faker
import random

# Seed for reproducibility
random.seed(42)

# Create a Faker instance
fake = Faker()

# Generate synthetic data
data = {
    'Full Name': [fake.name() for _ in range(1000)],
    'Probability': [random.uniform(0, 1) for _ in range(1000)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Extract first name, middle name, and surname
df[['First Name', 'Middle Name', 'Surname']] = df['Full Name'].str.split(n=2, expand=True)

# Handle the case where a name might not have a middle name
df['Middle Name'].fillna('', inplace=True)

# Replace None values in the 'Surname' column with an empty string
df['Surname'].fillna('', inplace=True)

# Generate nicknames based on the first name, middle name, and surname
def generate_nicknames(row):
    nicknames = [row['First Name']]
    if row['Middle Name']:
        nicknames.append(row['Middle Name'][:1])
    nicknames.append(row['Surname'][:3])
    return nicknames

df['Nicknames'] = df.apply(generate_nicknames, axis=1)

# Save the DataFrame to a CSV file
df.to_csv('your_dataset.csv', index=False)
