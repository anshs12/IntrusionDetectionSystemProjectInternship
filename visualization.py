import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px

# Load dataset
filepath = "CICIDS2017Dataset\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Display basic info about dataset
print(df.info())
print(df.head())
print(df.describe())

# Check column names
print(df.columns)

# Replace infinity values with NaNs and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Assume 'Label' is the target column (change it if needed)
label_column = 'Label'

if label_column not in df.columns:
    raise KeyError(f"'{label_column}' not found in the dataset columns: {df.columns}")

# Encode labels
le = LabelEncoder()
df[label_column] = le.fit_transform(df[label_column])

# Standardize numerical features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define features (X) and target (y)
X = df.drop(label_column, axis=1)
y = df[label_column]

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support(indices=True)]
print(selected_features)

# Select features for linear regression (change 'selected_features' as needed)
features_for_regression = selected_features[:5]  # Example: select the first 5 features
X_reg = df[features_for_regression]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Define features (X) and target (y)
X = df[['Total Fwd Packets']]
y = df['Total Backward Packets']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Scatter plot with regression line using Seaborn with enhanced styling
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.regplot(x='Total Fwd Packets', y='Total Backward Packets', data=df, scatter_kws={'alpha':0.6}, line_kws={'color': 'red', 'linewidth': 2})
plt.title('Total Forward vs Backward Packets with Linear Regression Line', fontsize=16)
plt.xlabel('Total Forward Packets', fontsize=14)
plt.ylabel('Total Backward Packets', fontsize=14)

# Example annotation
max_point = df.loc[df['Total Fwd Packets'].idxmax()]
plt.annotate('Max Total Fwd Packets', xy=(max_point['Total Fwd Packets'], max_point['Total Backward Packets']),
             xytext=(max_point['Total Fwd Packets']*1.05, max_point['Total Backward Packets']*1.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Add text box with R^2 and MSE
textstr = f'R²: {r2:.2f}\nMSE: {mse:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=props)

plt.show()

# Scatter plot with regression line using Matplotlib with enhanced styling
plt.figure(figsize=(12, 8))
plt.scatter(df['Total Fwd Packets'], df['Total Backward Packets'], alpha=0.6, label='Data Points')
plt.plot(df['Total Fwd Packets'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Total Forward vs Backward Packets with Linear Regression Line', fontsize=16)
plt.xlabel('Total Forward Packets', fontsize=14)
plt.ylabel('Total Backward Packets', fontsize=14)
plt.legend()
plt.grid(True)

# Add text box with R^2 and MSE
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=props)

plt.show()
