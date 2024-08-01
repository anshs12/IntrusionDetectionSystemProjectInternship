import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset (adjust the path as per your setup)
filepath = "CICIDS2017Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Display basic info about the dataset
print(df.info())
print(df.head())
print(df.describe())

# Replace infinity values with NaNs and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define features (X) and target (y)
features = ['Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Destination Port']  # Add relevant features
target = 'Label'  # Assuming 'Label' column indicates attack or benign

X = df[features]
y = df[target]

# Map 'BENIGN' to 0 and other labels (e.g., 'DDoS') to 1
y = y.apply(lambda x: 0 if x == 'BENIGN' else 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
