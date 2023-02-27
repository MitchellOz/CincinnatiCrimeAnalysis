import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

crime_data = pd.read_csv('crime_data.csv')

X = crime_data[['predictor_variable_1', 'predictor_variable_2', ...]]
y = crime_data['target_variable']

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Mean squared error:", mse)
print("R-squared:", r2)
