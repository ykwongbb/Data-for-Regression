import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

sample_size_2022 = 6 #have only 10 months
sample_size_2021 = 8 #have all 12 months
df = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/1M-3M.csv', header = None)
df1M_3M = pd.read_csv('/Users/anson/Library/CloudStorage/OneDrive-HKUSTConnect/Yr2 summer/Urop1000/FilesSumUp0206/Data for Regression/1M-3M.csv')

def bullish_or_bearish(shift, no_of_months, sample_size):
    for j in range(len(df.iloc[0]) - 2):
        count = 0
        col_1 = df1M_3M[df.iloc[0][j + 2]].values
        x = np.array([])
        y = np.array([])
        for i  in range(no_of_months): 
            if (pd.isna(col_1[i + shift]) == False):  
                if (col_1[i + shift] < 0):
                    x = np.append(x, col_1[i])
                elif (col_1[i + shift] > 0):
                    y = np.append(y, col_1[i])
                count += 1
        if (count >= sample_size):
            if (len(x) >= count * 2 / 3):
                print(df.iloc[0][j + 2], 'bearish')
            elif (len(y) >= count * 2 / 3):
                print(df.iloc[0][j + 2], 'bullish')
print('1M-3M')
print('2022 ')
bullish_or_bearish(0, 10, sample_size_2022)
print('-----------------------------')
print('2021')
bullish_or_bearish(10, 12, sample_size_2021)
print('-----------------------------')
print('2020')
bullish_or_bearish(22, 12, sample_size_2021)
print('-----------------------------')
print('2019')
bullish_or_bearish(34, 12, sample_size_2021)
print('-----------------------------')
print('2018')
bullish_or_bearish(46, 12, sample_size_2021)