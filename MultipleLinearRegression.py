import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Loading
file_path = '/Users/lukassspazo/Year 3 Python/AIoT/Tutorials/Autoencoder For Trading/insurance.csv'
df = pd.read_csv(file_path)

#Debug
print("Data Preview:")
print(df.head())

#Preprocess categorical variables to dummy variables
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#Print the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

#Predictions
y_pred = model.predict(X_test)

#Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R² Score: {r2}')

#Residuals vs Fitted plot
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.show()

#Normal Q-Q plot
plt.figure(figsize=(10, 6))
sm.qqplot(y_test - y_pred, line='s')
plt.title('Normal Q-Q')
plt.show()