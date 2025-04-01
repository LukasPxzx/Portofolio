import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflowjs as tfjs
import json
import os
import shutil

#USE PYTHON 3.9.2

#RCP scenarios for stochastic approach and seasonn differentation
RCP_MAPPING = {
    'results_RCP2_6.csv': 0,
    'results_RCP4_5.csv': 1,
    'results_RCP8_5.csv': 2
}

#Function for Loading data from multiple CSV files and combining them with an RCP column
def load_data(file_paths):
    data_list = []
    for file_path in file_paths:
        rcp_scenario = os.path.basename(file_path)
        data = pd.read_csv(file_path)
        data['RCP'] = RCP_MAPPING[rcp_scenario] 
        data_list.append(data)
    
    combined_data = pd.concat(data_list, ignore_index=True)


    print("Columns in combined_data:", combined_data.columns)
    print("First few rows of combined_data:\n", combined_data.head())
    
    return combined_data

#Preprocessing
def preprocess_data(data):
    #Create binary mold growth indicator
    data['Mold_Growth'] = data.apply(lambda row: 1 if (row['RH2M'] > 60 and 20 <= row['TS'] <= 30) else 0, axis=1)

    #Prepare features and apply scaling
    features = data[['RH2M', 'TS', 'RCP']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
  
    X, y = [], []
    sequence_length = 10
    for i in range(len(scaled_features) - sequence_length - 50):
        X.append(scaled_features[i:i + sequence_length])
        y.append(data['M'].iloc[i + sequence_length + 50])  #target

    return np.array(X), np.array(y), scaler

#Final Load
file_paths = [
    '/Users/lukassspazo/Year 3 Python/IoT/Final Project/results_RCP8_5.csv',
    '/Users/lukassspazo/Year 3 Python/IoT/Final Project/results_RCP4_5.csv',
    '/Users/lukassspazo/Year 3 Python/IoT/Final Project/results_RCP2_6.csv'
]
data = load_data(file_paths)
X, y, scaler = preprocess_data(data)

#SPLITS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#RNN model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Input layer specifying the shape
model.add(LSTM(100, return_sequences=True))  # LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

#Custom LR + OPT
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

#Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

#Early stopping + History for loss
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

#Evaluation
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test MAE: {mae:.4f}')

#Plots
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Save model in TensorFlow.js format
save_path = '/Users/lukassspazo/Year 3 Python/IoT/Final Project/static/models/MoldGrowthPrediction/tfjs_model'

#Remove existing directory if it exists
if os.path.exists(save_path):
    shutil.rmtree(save_path)

#Build the model to ensure input shape is correct
model.build((None, X_train.shape[1], X_train.shape[2]))  # Reminder, comment this :0

#Save the model
tfjs.converters.save_keras_model(model, save_path)

#Nota: No separate metadata file is needed; TensorFlow.js should handle it within model.json.

#Function to predict M after days
def predict_m_after_days(input_sequence, days_ahead):
    if input_sequence.shape[1] != 3:  #Make sure input shape matches expected (n, 3)
        raise ValueError("Input sequence must have the shape (n, 3) where n is the number of time steps.")
    
    input_data = scaler.transform(input_sequence)
    
    #Prepare the initial sequence for prediction
    sequence = np.array([input_data])
    
    predictions = []
    
    for _ in range(days_ahead):
        
        pred = model.predict(sequence)
        predictions.append(pred[0][0])
        new_input_data = np.array([[pred[0][0], input_data[-1][1], input_data[-1][2]]])  #Keep last humidity and RCP
        sequence = np.append(sequence[:, 1:, :], new_input_data.reshape(1, 1, 3), axis=1)
        input_data = np.append(input_data, new_input_data, axis=0) 

    return predictions

#EJEMPLO DEBUG
input_sequence = np.array([
    [97.62, 2.48, 0],  # Humidity (RH2M), Temperature (TS), RCP
    [96.62, 3.64, 0],
    [95.00, 3.00, 0],
    [94.00, 2.00, 0],
    [93.50, 3.50, 0],
    [92.00, 3.00, 0],
    [91.00, 2.50, 0],
    [90.00, 2.00, 0],
    [89.00, 2.75, 0],
    [88.00, 3.00, 0],
])  

#Predict the next 50 days
predicted_m_values = predict_m_after_days(input_sequence, 50)

# Print the predicted values
print("Predicted M values for the next 50 days:", predicted_m_values)

# Visualize predicted values
plt.figure(figsize=(12, 6))
plt.plot(range(50), predicted_m_values, label='Predicted M', color='red')
plt.title('Predicted Mold Growth Index Over 50 Days')
plt.xlabel('Days')
plt.ylabel('Predicted M Value')
plt.legend()
plt.show()