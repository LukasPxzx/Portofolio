import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

#Loading
file_path = '/Users/lukassspazo/Year 3 Python/AIoT/Tutorials/Autoencoder For Trading/insurance.csv'
df = pd.read_csv(file_path)

#Preprocessing of Categorical Variables
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

#Isolation Forest
features = df.drop('charges', axis=1)  # Exclude target variable
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['predicted_anomaly'] = iso_forest.fit_predict(features)

#Convert predicted anomalies to binary (1 for anomaly, 0 for normal)
df['predicted_anomaly'] = np.where(df['predicted_anomaly'] == -1, 1, 0)

#Analyze Anomalies
anomalies = df[df['predicted_anomaly'] == 1]
print(f"Number of anomalies detected: {len(anomalies)}")
print("Anomalies found:\n", anomalies)

#Results
plt.figure(figsize=(15, 10))

#Age vs. BMI
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='age', y='bmi', hue='predicted_anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.title('Age vs. BMI')

#Age vs. Charges
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='age', y='charges', hue='predicted_anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.title('Age vs. Charges')

#BMI vs. Charges
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='bmi', y='charges', hue='predicted_anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.title('BMI vs. Charges')

#Children vs. Charges
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='children', y='charges', hue='predicted_anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.6)
plt.title('Children vs. Charges')

plt.tight_layout()
plt.show()