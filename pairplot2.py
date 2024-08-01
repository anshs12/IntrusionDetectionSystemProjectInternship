import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Selected features for pair plot
selected_features = [
    'Total Fwd Packets',
    'Flow IAT Max',
    'Fwd IAT Total',
    'Label_cat'
]

# Filter data for benign and DDoS
df_benign = df[df['Label_cat'] == 0]
df_ddos = df[df['Label_cat'] == 1]

# Create pair plot for benign traffic
g_benign = sns.pairplot(df_benign[selected_features[:-1]], plot_kws={'alpha': 0.6, 's': 10})
g_benign.fig.suptitle('Pair Plot of Selected Features (Benign)', y=1.02)
plt.show()

# Create pair plot for DDoS traffic
g_ddos = sns.pairplot(df_ddos[selected_features[:-1]], plot_kws={'alpha': 0.6, 's': 10})
g_ddos.fig.suptitle('Pair Plot of Selected Features (DDoS)', y=1.02)
plt.show()
