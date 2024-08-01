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

# Filter dataset to only include DDoS points
ddos_df = df[df['Label'] == 'DDoS']

# Selected features for pair plot
selected_features = [
    'Total Fwd Packets',
    'Flow IAT Max',
    'Fwd IAT Total',
    'Label_cat'
]

# Custom color palette for DDoS only
custom_palette_ddos = {1: 'red'}  # color for DDoS traffic

# Pair plot with DDoS data only
g = sns.pairplot(ddos_df[selected_features], hue='Label_cat', palette=custom_palette_ddos, plot_kws={'alpha': 0.6, 's': 10})

# Plot title and labels
g.fig.suptitle('Pair Plot of Selected Features (DDoS Only)', y=1.02)

# Optionally, adjust plot size
g.fig.set_size_inches(10, 8)

plt.show()
