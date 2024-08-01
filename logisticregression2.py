import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
filepath = "CICIDS2017Dataset\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Clean the dataset (remove NaNs and infinities)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Encode categorical labels
df['Label'] = df['Label'].astype('category')
df['Label_cat'] = df['Label'].cat.codes  # Create a new column with categorical codes

# Define selected features
features = [
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Duration',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

X = df[features]
y = df['Label_cat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline with StandardScaler, SelectKBest, and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=20)),  # Select top 20 features
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = pipeline.predict_proba(X_test)

# Evaluate the model
y_pred = pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot ROC curve for each class
plt.figure(figsize=(12, 8))
for i in range(len(df['Label'].cat.categories)):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {df["Label"].cat.categories[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Probability of DDoS Attack vs. Total Forward Packets
ddos_label = df['Label'].cat.categories.get_loc('DDoS')  # Get the integer code for DDoS
y_test_ddos = (y_test == ddos_label).astype(int)

# Scale the single feature
x_values = X_test['Total Fwd Packets'].values.reshape(-1, 1)
single_feature_scaler = StandardScaler()
x_values_scaled = single_feature_scaler.fit_transform(x_values)

# Predict probabilities for the DDoS class
y_pred_proba_ddos = pipeline.predict_proba(X_test)[:, ddos_label]

plt.figure(figsize=(12, 8))

# Scatter plot of actual data points
plt.scatter(X_test['Total Fwd Packets'], y_test_ddos, color='blue', label='Actual', alpha=0.6, edgecolor='k')

# Logistic regression curve
plt.plot(X_test['Total Fwd Packets'], y_pred_proba_ddos, color='red', linewidth=2, label='Logistic Regression')



# Visualization: Probability of Other Attacks vs. Total Forward Packets
attack_labels = [label for label in df['Label'].cat.categories if label != 'DDoS']

for attack in attack_labels:
    attack_label = df['Label'].cat.categories.get_loc(attack)
    y_test_attack = (y_test == attack_label).astype(int)
    y_pred_proba_attack = pipeline.predict_proba(X_test)[:, attack_label]

    plt.figure(figsize=(12, 8))

    # Scatter plot of actual data points
    plt.scatter(X_test['Total Fwd Packets'], y_test_attack, color='blue', label='Actual', alpha=0.6, edgecolor='k')

    # Logistic regression curve
    plt.plot(X_test['Total Fwd Packets'], y_pred_proba_attack, color='red', linewidth=2, label='Logistic Regression')


