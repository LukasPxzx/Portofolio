# Stock Price Prediction Using Machine Learning and Deep Learning

## Overview

This project aims to predict stock prices using various machine learning (ML) and deep learning (DL) algorithms. It implements models such as Linear Regression, Random Forest, and Long Short-Term Memory (LSTM) networks to analyze historical stock data and make predictions.

## Features

- **Linear Regression**: A simple approach to predict stock prices based on historical data.
- **Random Forest Regressor**: An ensemble method that improves prediction accuracy by averaging multiple decision trees.
- **LSTM**: A deep learning model specifically designed for sequence prediction, beneficial for time series data.

## Installation

### Prerequisites

- Python 3.7 or higher
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Keras

### Required Libraries

Install the necessary Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras
```

## Data Overview

The dataset used in this analysis contains historical stock prices, including adjusted close prices and trading volumes. The goal is to predict future stock prices based on past performance.

## Analysis Steps

### Loading the Data

The dataset is loaded from a CSV file. Ensure the file path is correct.

```python
file_path = '/Users/lukassspazo/Year 3 Python/AIoT/Tutorials/Autoencoder For Trading/dataMSFTday2024.csv'
df = pd.read_csv(file_path)
```

### Data Preprocessing

- Convert the 'Date' column to datetime format and set it as the index.
- Create lagged features and the target variable for prediction.
- Clean the dataset by removing rows with missing values.

```python
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Adj Close', 'Volume']].copy()
data['Prev Adj Close'] = data['Adj Close'].shift(1)
data['Target'] = data['Adj Close'].shift(-1)
data.dropna(inplace=True)
```

### Feature Scaling

Normalize the features using MinMaxScaler for better performance of the models.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[['Prev Adj Close', 'Volume']])
```

### Splitting the Data

Split the data into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['Target'], test_size=0.2, random_state=42)
```

### Model Training

- **Linear Regression**: Train and evaluate the model.
- **Random Forest Regressor**: Train and evaluate the model.
- **LSTM**: Reshape the data, build, and train the model.

```python
# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Reshape data for LSTM
X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
```

### Predictions and Evaluations

Make predictions and evaluate all models using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

```python
y_pred_lin = lin_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test_lstm)
```

### Visualizations

Generate various visualizations to analyze the model performance, including residuals and loss functions.

```python
# Residuals for Linear Regression
plt.figure(figsize=(14, 5))
plt.scatter(y_test, y_pred_lin - y_test, color='blue', label='Residuals', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
plt.title('Linear Regression Residuals')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.legend()
plt.grid()
plt.show()
```

## Running the Analysis

Ensure the dataset is in the correct path. Run the scripts:

```bash
python stock_price_prediction.py
```

## Usage

- **Data Visualization**: Visual aids to understand model performance and predictions.
- **Model Interpretation**: Analyze the predictions and residuals from the models.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or enhancements.

## Acknowledgments

- Scikit-learn for machine learning tools.
- Keras for deep learning methodologies.
- Matplotlib and Seaborn for data visualization.
