import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Load the dataset
filepath = "CICIDS2017Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Replace infinity values with NaNs and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Filter the dataframe to only contain relevant target classes
df = df[df['Label'].isin(['BENIGN', 'DDoS'])]

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

# SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Fit the model
svm_model.fit(X_train, y_train)

# Function to plot decision boundaries with labeled points
def plot_decision_boundary(clf, X, y, title, feature_names):
    # Setting limits for x and y axes
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Creating a grid of points for prediction
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Predicting the class for each grid point
    Z = Z.reshape(xx.shape)  # Reshape Z to the shape of xx
    plt.figure(figsize=(10, 6))  # Specifying the size of the plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(('blue', 'red')))  # Contour plot for decision boundary
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=ListedColormap(('blue', 'red')))  # Plotting data points
    plt.title(title)  # Title for the plot
    plt.xlabel(feature_names[0] + ' (PCA)')  # Label for x-axis
    plt.ylabel(feature_names[1] + ' (PCA)')  # Label for y-axis
    
    # Create legend for the plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='BENIGN'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='DDoS')]
    plt.legend(handles=handles, loc='best')
    
    plt.show()  # Display the plot

# Plot decision boundary using PCA-reduced features
plot_decision_boundary(svm_model, X_test, y_test, "Decision Boundary for SVM Model (PCA-reduced)", ['PC1', 'PC2'])
