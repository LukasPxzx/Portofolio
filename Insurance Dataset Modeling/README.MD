# Insurance Dataset Analysis with Linear Regression and Anomaly Detection

## Overview

This project analyzes an insurance dataset using various techniques, including Multiple Linear Regression, Autoencoder for anomaly detection, Isolation Forest, Q-Q plots, and Frequency Analysis. The goal is to understand the factors influencing insurance charges and identify anomalies within the dataset using advanced machine learning methodologies.

## Features

- **Multiple Linear Regression**: Predicts insurance charges based on various factors.
- **Autoencoder for Anomaly Detection**: Identifies outliers by reconstructing input data and calculating reconstruction errors.
- **Isolation Forest**: Detects anomalies using a tree-based ensemble method.
- **Q-Q Plot**: Assesses the normality of residuals from the regression model.
- **Frequency Analysis**: Analyzes categorical variables for deeper insights.

## Installation

### Prerequisites

- Python 3.7 or higher
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow

### Required Libraries

Install the necessary Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## Data Overview

The dataset used in this analysis contains various features related to individuals' insurance claims, including demographic information and health-related factors. It helps in predicting insurance charges and detecting anomalies.

## Analysis Steps

### Loading the Data

The dataset is loaded from a CSV file. Ensure the file path is correct.

```python
file_path = '/path/to/insurance.csv'
df = pd.read_csv(file_path)
```

### Data Preprocessing

Categorical variables are converted into dummy variables for regression analysis. The features used in the Autoencoder and Isolation Forest exclude categorical variables and are scaled using Standard Scaler.

### Multiple Linear Regression

A Multiple Linear Regression model is trained on the training data to predict insurance charges. The model's performance is evaluated using Mean Squared Error (MSE) and R² Score.

### Autoencoder for Anomaly Detection

An Autoencoder is implemented to detect anomalies by reconstructing input data and calculating reconstruction errors. The steps include:

- **Data Normalization**: Features are standardized for better performance.
- **Model Architecture**: The Autoencoder consists of an encoder and decoder with L2 regularization.
- **Training**: The model is trained with early stopping and learning rate reduction.
- **Anomaly Detection**: Reconstruction errors are calculated, and anomalies are identified based on a set threshold.

### Isolation Forest

An Isolation Forest model is employed to detect anomalies based on the features of the dataset. Anomalies are identified and visualized in relation to key variables, such as age, BMI, and charges.

### Visualization

The scripts generate various visualizations, including:

- Loss Function Over Epochs: Shows training and validation loss during Autoencoder training.
- Feature Contributions: Bar plots display reconstruction errors for different features.
- Correlation Matrix: Visualizes correlations among key features.
- Anomaly Detection Visualizations: Scatter plots illustrate relationships between features and insurance charges, highlighting detected anomalies.
- Frequency Graphs: Analyzes categorical variables such as smoker status and sex.
- Key Features vs Charges: Scatter plots illustrating relationships between features and insurance charges.

## Running the Analysis

Ensure the dataset is in the correct path. Run the scripts:

```bash
python MultipleLinearRegression.py
python AutoencoderSTANDARDSCALER.py
python IsolationForest.py
```

## Usage

- **Data Visualization**: The scripts generate visualizations to aid in understanding model performance and assumptions.
- **Model Interpretation**: Review coefficients from regression and anomalies from the Autoencoder and Isolation Forest.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or enhancements.

## Acknowledgments

- Scikit-learn for machine learning tools.
- TensorFlow for deep learning methodologies.
- Statsmodels for statistical modeling.
- Matplotlib and Seaborn for data visualization.
