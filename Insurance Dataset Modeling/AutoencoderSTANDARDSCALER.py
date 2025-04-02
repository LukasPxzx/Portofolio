import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2  
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

#Loading
file_path = '/Users/lukassspazo/Year 3 Python/AIoT/Tutorials/Autoencoder For Trading/insurance.csv'
df = pd.read_csv(file_path)

#Exclude categorical variables
features = df[['age', 'bmi', 'children', 'charges']].values

#Since Q-Q is close to fitted line, use Standard Scaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(features)

#Data Split
train_size = int(len(data_normalized) * 0.8)
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

#Autoencoder with L2 regularization
input_dim = train_data.shape[1]
encoding_dim = 128  
hidden_dim1 = 64    
hidden_dim2 = 32    
l2_regularization = 0.01  

#Input
input_layer = Input(shape=(input_dim,))

#Encoder
encoder = Dense(hidden_dim1, activation='tanh', kernel_regularizer=l2(l2_regularization))(input_layer)
encoder = Dense(hidden_dim2, activation='tanh', kernel_regularizer=l2(l2_regularization))(encoder)
encoder = Dense(encoding_dim, activation='tanh', kernel_regularizer=l2(l2_regularization))(encoder)

#Decoder
decoder = Dense(hidden_dim2, activation='tanh', kernel_regularizer=l2(l2_regularization))(encoder)
decoder = Dense(hidden_dim1, activation='tanh', kernel_regularizer=l2(l2_regularization))(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

#Compilation
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#Early stopping and LR reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

#Training
history = autoencoder.fit(train_data, train_data,
                          epochs=500,  
                          batch_size=32,
                          shuffle=False,
                          validation_data=(test_data, test_data),
                          callbacks=[early_stopping, reduce_lr])

#Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Function Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Data Compression ( Feature Extraction )
encoder_model = Model(inputs=input_layer, outputs=encoder)
encoded_data = encoder_model.predict(test_data)

#Anomaly Detection (Calculate reconstruction error)
reconstructed_data = autoencoder.predict(test_data)
reconstruction_error = np.mean((test_data - reconstructed_data) ** 2, axis=1)

#Threshold for anomalous data
threshold = np.percentile(reconstruction_error, 95)
anomalies = np.where(reconstruction_error > threshold)[0]


anomaly_data = test_data[anomalies]
anomaly_df = pd.DataFrame(anomaly_data, columns=['Normalized Age', 'Normalized BMI', 'Normalized Children', 'Normalized Charges'])

# Debugging
print("Anomalies (Normalized):")
print(anomaly_df)

#Analysis, Keep Original Data
original_anomalies = df.iloc[train_size + anomalies]
print("\nOriginal Data Points for Anomalies:")
print(original_anomalies)

#Calculate the RE for each feature on normalized data
reconstruction_errors = np.abs(test_data[anomalies] - reconstructed_data[anomalies])

#Create a DataFrame to display the anomalies with errors
error_df = pd.DataFrame(reconstruction_errors, columns=['Age Error', 'BMI Error', 'Children Error', 'Charges Error'])
anomaly_analysis = pd.concat([original_anomalies.reset_index(drop=True), error_df], axis=1)

#Debug
print("\nAnomalies with Reconstruction Errors:")
print(anomaly_analysis)

total_errors = reconstruction_errors.sum(axis=0)

#Bar plot for total reconstruction errors by feature
plt.figure(figsize=(10, 6))
feature_names = ['Age Error', 'BMI Error', 'Children Error', 'Charges Error']
plt.bar(feature_names, total_errors, color=['blue', 'orange', 'green', 'red'])
plt.title('Total Reconstruction Errors by Feature for Anomalies')
plt.xlabel('Features')
plt.ylabel('Total Reconstruction Error')
plt.show()

#Feature contributions for anomalies
plt.figure(figsize=(12, 8))
sns.barplot(data=anomaly_analysis.melt(id_vars=['age', 'bmi', 'children', 'charges'],
                                         value_vars=['Age Error', 'BMI Error', 'Children Error']),
             x='variable', y='value', hue='charges', palette='Set1')
plt.title('Reconstruction Errors of Anomalous Charges by Feature')
plt.xlabel('Feature')
plt.ylabel('Reconstruction Error')
plt.xticks(rotation=45)
plt.legend(title='Anomalous Charges')
plt.show()

#Correlation matrix
correlation_matrix = anomaly_analysis[['age', 'bmi', 'children', 'charges']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix')
plt.show()

#Encoded data and anomalies
plt.figure(figsize=(10, 6))
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], label='Encoded Data', alpha=0.6)
plt.scatter(encoded_data[anomalies, 0], encoded_data[anomalies, 1], color='red', label='Anomalies', alpha=0.8)

#Encoded data (excluding anomalies)
x = encoded_data[:, 0]
y = encoded_data[:, 1]

#Linear Fit
coefficients = np.polyfit(x, y, deg=1)
poly = np.poly1d(coefficients)
x_fit = np.linspace(min(x), max(x), 100)
plt.plot(x_fit, poly(x_fit), color='green', linestyle='--', label='Fitting Line')
plt.title('Encoded Data with Anomalies and Fitting Line')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.legend()
plt.show()

#Reconstruction Error Analysis: Visualizing the RE
plt.figure(figsize=(10, 6))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.title('Reconstruction Error')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.legend()
plt.show()

#Frequency Graphs
plt.figure(figsize=(15, 10))

#Smoker Status
plt.subplot(2, 2, 1)
sns.countplot(x='smoker', data=df, palette='Set2')
plt.title('Count of Smokers vs Non-Smokers')
plt.xlabel('Smoker Status')
plt.ylabel('Count')

#Sex
plt.subplot(2, 2, 2)
sns.countplot(x='sex', data=df, palette='Set2')
plt.title('Count of Male vs Female')
plt.xlabel('Sex')
plt.ylabel('Count')

#Region
plt.subplot(2, 2, 3)
sns.countplot(x='region', data=df, palette='Set2')
plt.title('Count of Clients by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#Charges Graphs
plt.figure(figsize=(15, 10))

#Average Charges by Sex
plt.subplot(2, 2, 1)
sns.barplot(x='sex', y='charges', data=df, estimator=np.mean, palette='Set1')
plt.title('Average Charges by Sex')
plt.xlabel('Sex')
plt.ylabel('Average Charges')

#Average Charges by Smoker Status
plt.subplot(2, 2, 2)
sns.barplot(x='smoker', y='charges', data=df, estimator=np.mean, palette='Set1')
plt.title('Average Charges by Smoker Status')
plt.xlabel('Smoker Status')
plt.ylabel('Average Charges')

#Average Charges by Region
plt.subplot(2, 2, 3)
sns.barplot(x='region', y='charges', data=df, estimator=np.mean, palette='Set1')
plt.title('Average Charges by Region')
plt.xlabel('Region')
plt.ylabel('Average Charges')

plt.tight_layout()
plt.show()

# Visualization of Key Features vs Charges
plt.figure(figsize=(15, 10))

#Age vs Charges
plt.subplot(3, 1, 1)
sns.scatterplot(x=df['age'], y=df['charges'], alpha=0.6)
plt.scatter(df['age'][anomalies], df['charges'][anomalies], color='red', label='Anomalies', alpha=0.8)
plt.title('Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()

#BMI vs Charges
plt.subplot(3, 1, 2)
sns.scatterplot(x=df['bmi'], y=df['charges'], alpha=0.6)
plt.scatter(df['bmi'][anomalies], df['charges'][anomalies], color='red', label='Anomalies', alpha=0.8)
plt.title('BMI vs Charges')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.legend()

#Children vs Charges
plt.subplot(3, 1, 3)
sns.scatterplot(x=df['children'], y=df['charges'], alpha=0.6)
plt.scatter(df['children'][anomalies], df['charges'][anomalies], color='red', label='Anomalies', alpha=0.8)
plt.title('Children vs Charges')
plt.xlabel('Children')
plt.ylabel('Charges')
plt.legend()

plt.tight_layout()
plt.show()