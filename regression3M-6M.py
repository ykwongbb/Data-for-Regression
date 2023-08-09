import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

sample_size = 15
df = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/1M-3M.csv', header = None)
df3M_6M = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/3M-6M.csv')
dfActual_3M = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/Actual-3M.csv')

count = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_6 = 0
slope = np.array([])
t_stat = np.array([])
p_value = np.array([])
R_square = np.array([])

for j in range(len(df.iloc[0]) - 2):
    col_1 = df3M_6M[df.iloc[0][j + 2]].values.reshape(-1, 1)
    col_2 = dfActual_3M[df.iloc[0][j + 2]].values.reshape(-1, 1)
    x = np.array([])
    y = np.array([])

    for i in range(len(col_1) - 2):
        if (pd.isna(col_1[i][0]) == False) and (pd.isna(col_2[i][0]) == False):
            x = np.append(x, col_1[i][0]).reshape(-1, 1)
            y = np.append(y, col_2[i][0])
    if (len(y) >= sample_size):
        count_1 += 1
        reg = LinearRegression().fit(x, y)
        reg_pre = reg.predict(x)
        residuals = y - reg_pre
        print(df.iloc[0][j + 2])
        print('y-intercept:', reg.intercept_)
        print('Slope:', reg.coef_[0])
        slope = np.append(slope, reg.coef_[0])
        R_square = np.append(R_square, r2_score(y, reg_pre))

        print('MSE:', mean_squared_error(y, reg_pre))
        print("R-square:", r2_score(y, reg_pre))
        if (r2_score(y, reg_pre) > 0.2):
            count_2 += 1

        # Calculate the t-value and one-sided p-value using stats.t
        dof = len(x) - 2  # Degrees of freedom
        t = reg.coef_[0] / np.sqrt((np.sum(residuals**2) / (len(x) - 2)) / np.sum((x - np.mean(x))**2)) # Calculate t-value
        p = stats.t.sf(np.abs(t), dof)  # Calculate one-sided p-value
        t_stat = np.append(t_stat, t)
        p_value = np.append(p_value, p)
        if (p < 0.01):
            count_4 += 1
        if (p < 0.05):
            count_5 += 1
        if(p < 0.1):
            count_6 += 1
            
        elif (t > 2):
            count_3 += 1

plt.scatter(p_value, slope)
plt.ylabel('Coefficient of Regression')
plt.xlabel('p-value')
plt.title('Coefficient of Regression against p-value')
plt.axvline(x=0.01, color='red', linestyle='--', label='p-value = 0.01')
plt.axvline(x=0.05, color='blue', linestyle='--', label='p-value = 0.05')
plt.axvline(x=0.1, color='green', linestyle='--', label='p-value = 0.1')
plt.legend()
plt.show()

# Print the calculated p-values
for i in range(len(p_value)):
    print("Slope:", slope[i])
    print("T-Value:", t_stat[i])
    print("P-Value:", p_value[i])
    print("-------------------")
print("p < 0.01:", count_4)
print("p < 0.05:", count_5)
print("p < 0.1:", count_6)
  

# # Fit a quadratic curve to the data
# coeffs = np.polyfit(slope, R_square, 2)
# quad_fit = np.poly1d(coeffs)

# # Generate points along the quadratic curve
# x_quad = np.linspace(slope.min(), slope.max(), 100)
# y_quad = quad_fit(x_quad)

# # Plot the quadratic curve
# plt.plot(x_quad, y_quad, 'r')

# # Set the axis labels
# plt.xlabel('Slope')
# plt.ylabel('R-squared')

# # Set the plot title
# plt.title('3M-6M: Slope against R-square')

# print('-------------------------------------')

# # Calculate the residual sum of squares
# y_pred = quad_fit(slope)
# RSS = np.sum((R_square - y_pred)**2)

# # Calculate the total sum of squares
# TSS = np.sum((R_square - np.mean(R_square))**2)

# # Calculate the R-squared
# R2 = 1 - (RSS / TSS)

# # Print the RSS and R-squared
# print('RSS:', RSS)
# print('R-squared:', R2)
# # Show the plot
# plt.show()

#     #     x = np.delete(x, np.s_[:])
#     #     y = np.delete(y, np.s_[:])
#     # else:
#     #     x = np.delete(x, np.s_[:])
#     #     y = np.delete(y, np.s_[:]) 

# print(count_2, count_1, count)

