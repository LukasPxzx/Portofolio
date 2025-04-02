import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Loading
file_path = '/Users/lukassspazo/Year 3 Python/AIoT/Tutorials/Autoencoder For Trading/dataMSFTday2024.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)  # Use Date as Index

data = df[['Adj Close', 'Volume']].copy()
data['Prev Adj Close'] = data['Adj Close'].shift(1)  # Lagged Features
data['Target'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)  # Cleaning

X = data[['Prev Adj Close', 'Volume']]
y = data['Target']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test)

# Evaluation
lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_mae = mean_absolute_error(y_test, y_pred_lin)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_reg.predict(X_test)

# Evaluation
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)

# Train Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)

# Predictions
y_pred_poly = poly_reg.predict(X_poly_test)

# Evaluation
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_mae = mean_absolute_error(y_test, y_pred_poly)

# Train SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train, y_train)

# Predictions
y_pred_svr = svr_reg.predict(X_test)

# Evaluation
svr_mse = mean_squared_error(y_test, y_pred_svr)
svr_mae = mean_absolute_error(y_test, y_pred_svr)

# Train KNN
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

# Predictions
y_pred_knn = knn_reg.predict(X_test)

# Evaluation
knn_mse = mean_squared_error(y_test, y_pred_knn)
knn_mae = mean_absolute_error(y_test, y_pred_knn)

# Train Gradient Boosting Regressor
gbm_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm_reg.fit(X_train, y_train)

# Predictions
y_pred_gbm = gbm_reg.predict(X_test)

# Evaluation
gbm_mse = mean_squared_error(y_test, y_pred_gbm)
gbm_mae = mean_absolute_error(y_test, y_pred_gbm)

# ARIMA
# Prepare data for ARIMA
y_train_arima = data['Target'][:-len(y_test)]
y_train_arima.index = data.index[:-len(y_test)]

# Ensure the index is a datetime index
if not pd.api.types.is_datetime64_any_dtype(y_train_arima.index):
    y_train_arima.index = pd.to_datetime(y_train_arima.index)

# Create a date range with daily frequency if necessary
y_train_arima = y_train_arima.asfreq('D')

# Fit ARIMA model (change order as needed, here (1, 1, 1) is just an example)
arima_model = ARIMA(y_train_arima, order=(1, 1, 1))
arima_fit = arima_model.fit()

# Predictions
y_pred_arima = arima_fit.forecast(steps=len(y_test))

# Check if any NaN values are produced
if np.any(np.isnan(y_pred_arima)):
    print("Warning: ARIMA predictions contain NaN values.")
else:
    # Align the predictions with the test set index
    y_pred_arima_indexed = pd.Series(y_pred_arima, index=y_test.index)

    # Evaluation
    arima_mse = mean_squared_error(y_test, y_pred_arima_indexed)
    arima_mae = mean_absolute_error(y_test, y_pred_arima_indexed)

# Reshape the data for LSTM
X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the improved LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, activation='relu'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

# Compile
lstm_model.compile(optimizer='adam', loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# Fit model with early stopping and validation split
history = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=8, validation_split=0.2,
                         callbacks=[early_stopping], verbose=0)

# Predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Evaluation
lstm_mse = mean_squared_error(y_test, y_pred_lstm)
lstm_mae = mean_absolute_error(y_test, y_pred_lstm)

# Function to plot residuals
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(14, 5))
    plt.scatter(y_true, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
    plt.title(f'{model_name} Residuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid()
    plt.show()

# Plot residuals for each model
plot_residuals(y_test, y_pred_lin, 'Linear Regression')
plot_residuals(y_test, y_pred_rf, 'Random Forest')
plot_residuals(y_test, y_pred_poly, 'Polynomial Regression')
plot_residuals(y_test, y_pred_svr, 'SVR')
plot_residuals(y_test, y_pred_knn, 'KNN')
plot_residuals(y_test, y_pred_gbm, 'GBM')
plot_residuals(y_test, y_pred_arima_indexed, 'ARIMA')
plot_residuals(y_test, y_pred_lstm, 'LSTM')

# Function to plot predicted vs actual
def plot_predicted_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(14, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--', label='Ideal Prediction')
    plt.title(f'{model_name}: Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.grid()
    plt.show()

# Plot predicted vs actual for each model
plot_predicted_vs_actual(y_test, y_pred_lin, 'Linear Regression')
plot_predicted_vs_actual(y_test, y_pred_rf, 'Random Forest')
plot_predicted_vs_actual(y_test, y_pred_poly, 'Polynomial Regression')
plot_predicted_vs_actual(y_test, y_pred_svr, 'SVR')
plot_predicted_vs_actual(y_test, y_pred_knn, 'KNN')
plot_predicted_vs_actual(y_test, y_pred_gbm, 'GBM')
plot_predicted_vs_actual(y_test, y_pred_arima_indexed, 'ARIMA')
plot_predicted_vs_actual(y_test, y_pred_lstm, 'LSTM')
