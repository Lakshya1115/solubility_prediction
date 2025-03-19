import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

y = df['logS']

x = df.drop('logS', axis=1)
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(x, y, train_size = 0.2, random_state=100)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print("LR TRAIN MSE: ", lr_train_mse)
# print("LR TRAIN R2: ", lr_train_r2)
# print("LR TEST MSE: ", lr_test_mse)
# print("LR TEST R2: ", lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Method", "Train MSE", "Train R2", "Test MSE", "Test R2"]

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth = 2, random_state = 100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest Regression', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["Method", "Train MSE", "Train R2", "Test MSE", "Test R2"]

result_table = pd.concat([lr_results, rf_results], ignore_index = True, axis=0).reset_index(drop=True)
print(result_table)

import matplotlib.pyplot as plt
import numpy as np

## plt.figure(figsize = (5, 5))
plt.scatter(x = y_train, y = y_lr_train_pred, alpha=0.3, c = '#0000ff')

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.title('Linear Regression: Predicted vs. Experimental LogS', fontsize=16)
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')
plt.plot(y_train, p(y_train), '#ff0000')

import matplotlib.pyplot as plt
import numpy as np

## plt.figure(figsize = (5, 5))
plt.scatter(x = y_train, y = y_rf_train_pred, alpha=0.3, c = '#0000ff')

z = np.polyfit(y_train, y_rf_train_pred, 1)
p = np.poly1d(z)

plt.title('Random Forest: Predicted vs. Experimental LogS', fontsize=16)
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')
plt.plot(y_train, p(y_train), '#ff0000')

