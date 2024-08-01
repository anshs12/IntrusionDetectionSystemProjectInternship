import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the dataset
filepath = "CICIDS2017Dataset\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
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

# Label Encoding
df['Label'] = df['Label'].astype('category').cat.codes

# Define features (X) and target (y)
features = [
    'Destination Port'
]
X = df[features]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Decision Tree model with parameter tuning
model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize the decision tree structure
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=features, class_names=True, filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Structure')
plt.show()
