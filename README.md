# Urop 1000
## Linear regression of forecast error against forecast revision
1M - 3M, 3M - 6M, 6M - 1Y are referring to different time frames of the regression of different indivdual forecasters.
We set the minimum sample size to be 15 for each individual forecasters. Then we perform linear regression and find their respective p-value as well as the regression coefficient. After plotting the regression coefficient against p-value, we will know number of significant of regression coefficient we can look at. We further add p = 0.01, p = 0.05, p = 0.1 and print their respective number finally. There is also one doing regression on smart econimist by using the same condition to each individual forecasters as reference.

## Bullish or Bearish
Due to the imperfect of data, there are years not having 12 months so we use 2/3 * (the number of months the data) have to be the minimum sample size required. Then for each individual forecasters, we see whether they have negative or positive value in 1M - 3M, 3M - 6M, 6M - 1Y. Positive means bearish and negative means bullish. Since there are not many banks, we print all their names and views in the respective year