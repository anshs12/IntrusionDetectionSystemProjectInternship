import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import shap
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# Load the dataset
filepath = "CICIDS2017Dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Replace infinity values with NaNs and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Filter the dataframe to only contain 'BENIGN' and 'PortScan'
df = df[df['Label'].isin(['BENIGN', 'PortScan'])]

# Define features and target
features = [
    'Total Fwd Packets', 'Destination Port',
    'Flow Duration', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Bwd Packet Length Max',
    'Bwd Packet Length Mean', 'Packet Length Mean', 'Average Packet Size'
]
target = 'Label'

X = df[features]
y = df[target]

# Encode the target labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce the feature space to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=10,  # Reduced number of estimators
    verbosity=1,
    use_label_encoder=False
)

# Fit the model
xgb_model.fit(X_train, y_train)

# Ensure the model is trained
if xgb_model.get_booster().get_dump():
    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    # SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=['PC1', 'PC2'], plot_type="bar")

    # Detailed visualization for a single prediction
    instance_idx = 0
    shap.force_plot(explainer.expected_value, shap_values[instance_idx], X_test[instance_idx], feature_names=['PC1', 'PC2'], matplotlib=True)

    # Function to plot decision boundaries with labeled points
    def plot_decision_boundary(clf, X, y, title, feature_names):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(('blue', 'red')))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=ListedColormap(('blue', 'red')))
        plt.title(title)
        plt.xlabel(feature_names[0] + ' (PCA)')
        plt.ylabel(feature_names[1] + ' (PCA)')
        
        # Create legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='BENIGN'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='PortScan')]
        plt.legend(handles=handles, loc='best')
        
        plt.show()

    # Plot decision boundary using PCA-reduced features
    plot_decision_boundary(xgb_model, X_test, y_test, "Decision Boundary for XGBoost Model (PCA-reduced)", ['PC1', 'PC2'])
