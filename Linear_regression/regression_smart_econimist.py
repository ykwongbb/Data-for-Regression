import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/Smart_Economist_lin_reg.csv', header = None)
df_1 = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/Smart_Economist_lin_reg.csv')

col_1M_3M = df_1['1M-3M'].values.reshape(-1,1)
col_Actual_1M = df_1['Actual-1M'].values.reshape(-1,1)
col_3M_6M = df_1['3M-6M'].values.reshape(-1,1)
col_Actual_3M = df_1['Actual-3M'].values.reshape(-1,1)
col_6M_1Y = df_1['6M-1Y'].values.reshape(-1,1)
col_Actual_6M = df_1['Actual-6M'].values.reshape(-1,1)

reg_1 = LinearRegression().fit(col_1M_3M, col_Actual_1M)
reg_2 = LinearRegression().fit(col_3M_6M, col_Actual_3M)
reg_3 = LinearRegression().fit(col_6M_1Y, col_Actual_6M)

reg_pre_1 = reg_1.predict(col_1M_3M)
reg_pre_2 = reg_2.predict(col_3M_6M)
reg_pre_3 = reg_3.predict(col_6M_1Y)


print('y-intercept:', reg_1.intercept_)
print('Slope:', reg_1.coef_[0])
print('MSE:', mean_squared_error(col_Actual_1M, reg_pre_1))
print("R-square:", r2_score(col_Actual_1M, reg_pre_1))
residuals_1 = col_Actual_1M - reg_1.predict(col_1M_3M)
residual_std_error_1 = np.sqrt(np.sum(residuals_1 ** 2) / (len(col_1M_3M) - 2))
dof_1 = len(col_1M_3M) - 2
t_value = reg_1.coef_[0] / (residual_std_error_1 / np.sqrt(np.sum((col_1M_3M - np.mean(col_1M_3M)) ** 2)))
p = stats.t.sf(np.abs(t_value), dof_1) 

print('t_value:',p)
plt.scatter(col_1M_3M, col_Actual_1M)
plt.title('Smart Economist 1M-3M')
plt.xlabel('1M-3M')
plt.ylabel('Actual-1M')
plt.plot(col_1M_3M, reg_pre_1, color='red')
plt.show()
plt.close()
print('-----------------------------')
print('y-intercept:', reg_2.intercept_)
print('Slope:', reg_2.coef_[0])
print('MSE:', mean_squared_error(col_Actual_3M, reg_pre_2))
print("R-square:", r2_score(col_Actual_3M, reg_pre_2))
residuals_2 = col_Actual_3M - reg_2.predict(col_3M_6M)
residual_std_error_2 = np.sqrt(np.sum(residuals_2 ** 2) / (len(col_3M_6M) - 2))
dof_2 = len(col_3M_6M) - 2
t_value = reg_2.coef_[0] / (residual_std_error_2 / np.sqrt(np.sum((col_3M_6M - np.mean(col_3M_6M)) ** 2)))
p = stats.t.sf(np.abs(t_value), dof_2) 

print('t_value:',p)
plt.scatter(col_3M_6M, col_Actual_3M)
plt.title('Smart Economist 3M-6M')
plt.xlabel('3M-6M')
plt.ylabel('Actual-3M')
plt.plot(col_3M_6M, reg_pre_2, color='red')
plt.show()
plt.close()
print('-----------------------------')
print('y-intercept:', reg_3.intercept_)
print('Slope:', reg_3.coef_[0])
print('MSE:', mean_squared_error(col_Actual_6M, reg_pre_3))
print("R-square:", r2_score(col_Actual_6M, reg_pre_3))
residuals_3 = col_Actual_6M - reg_3.predict(col_6M_1Y)
residual_std_error_3 = np.sqrt(np.sum(residuals_3 ** 2) / (len(col_6M_1Y) - 2))
dof_3 = len(col_6M_1Y) - 2
t_value = reg_3.coef_[0] / (residual_std_error_3 / np.sqrt(np.sum((col_6M_1Y - np.mean(col_6M_1Y)) ** 2)))
p = stats.t.sf(np.abs(t_value), dof_3) 
print('t_value:',p)
plt.scatter(col_6M_1Y, col_Actual_6M)
plt.title('Smart Economist 6M-1Y')
plt.xlabel('6M-1Y')
plt.ylabel('Actual-6M')
plt.plot(col_6M_1Y, reg_pre_3, color='red')
plt.show()
plt.close()

