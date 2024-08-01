import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
filepath = "CICIDS2017Dataset\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(filepath)

# Clean dataset (remove NaNs and infinities)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Encode categorical labels
df['Label'] = df['Label'].astype('category')
df['Label_cat'] = df['Label'].cat.codes  # Create a new column with categorical codes

#selected features for pair plot
selected_features = [
  'Total Fwd Packets',
    'Flow IAT Max',
    'Fwd IAT Total',
    'Label_cat' 
]


# Custom color palette
custom_palette = {0: 'blue', 1: 'red'}  #colors for benign (0) and DDoS (1) traffic


#pair plot with smaller points and custom colors
g = sns.pairplot(df[selected_features], hue='Label_cat', palette=custom_palette, plot_kws={'alpha': 0.6, 's': 10})

#plot title and labels
g.fig.suptitle('Pair Plot of Selected Features (Benign vs DDoS)', y=1.02)
plt.show()
 