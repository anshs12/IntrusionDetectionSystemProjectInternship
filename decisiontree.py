import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset (adjust the path as per your setup)
filepath = "CICIDS2017Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Replace infinity values with NaNs and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define features (X) and target (y)
features = ['Total Fwd Packets', 'Destination Port']  # Add relevant features
target = 'Label'  # Assuming 'Label' column indicates attack or benign

X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Visualize the decision tree structure
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=features, class_names=model.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Distinguishing Benign vs. Attack Traffic')
plt.show()
