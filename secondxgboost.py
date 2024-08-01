import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb

# Load the dataset
filepath = "CICIDS2017Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Clean the dataset (remove NaNs and infinities)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Encode categorical labels
df['Label'] = df['Label'].astype('category')

# Mapping labels to more readable forms
label_mapping = {
    'BENIGN': 'Benign',
    'DDoS': 'DDoS',
    'Brute Force': 'Brute Force'
}
df['Label'] = df['Label'].map(label_mapping)

# Encode labels numerically
label_encoder = LabelEncoder()
df['Label_cat'] = label_encoder.fit_transform(df['Label'])

# Select relevant features for the pair plot
selected_features = [
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Duration',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Label'
]

# Standardize the selected features
scaler = StandardScaler()
df[selected_features[:-1]] = scaler.fit_transform(df[selected_features[:-1]])

# Create a pair plot
sns.set(style="whitegrid")
pair_plot = sns.pairplot(df[selected_features], hue='Label', palette='viridis', plot_kws={'alpha': 0.6, 's': 40})
pair_plot.fig.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

# Define features and target variable for XGBoost model
features = [
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Duration',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets'
]
X = df[features]
y = df['Label_cat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot ROC curve for each class
plt.figure(figsize=(12, 8))
for i, label in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (area = {roc_auc:.2f})')

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
