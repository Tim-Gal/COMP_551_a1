import numpy as np
import pandas as pd


def remove_outliers(df, iqr_factor=4):
    df_cleaned = df.copy()  # Create a copy of the original DataFrame
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned


bostondf = pd.read_csv("boston.csv")
bostondf_dropped = bostondf.dropna(axis=0, how='any', inplace=False)
bostondf_cleaned = remove_outliers(bostondf_dropped)


winedf = pd.read_csv("wine.data")
winedf_dropped = winedf.dropna(axis=0, how='any', inplace=False)
winedf_cleaned = remove_outliers(winedf_dropped)


mean_deviation_CRIM = bostondf['CRIM'].mean()
mean_deviation_ZN = bostondf['ZN'].mean()
mean_deviation_INDUS = bostondf['INDUS'].mean()
mean_deviation_CHAS = bostondf['CHAS'].mean()
mean_deviation_NOX = bostondf['NOX'].mean()
mean_deviation_RM = bostondf['RM'].mean()
mean_deviation_AGE = bostondf['AGE'].mean()
mean_deviation_DIS = bostondf['DIS'].mean()
mean_deviation_RAD = bostondf['RAD'].mean()
mean_deviation_TAX = bostondf['TAX'].mean()
mean_deviation_PTRATIO = bostondf['PTRATIO'].mean()
mean_deviation_LSTAT = bostondf['LSTAT'].mean()
mean_MEDV = bostondf['MEDV'].mean()

min_deviation_CRIM = bostondf['CRIM'].min()
min_deviation_ZN = bostondf['ZN'].min()
min_deviation_INDUS = bostondf['INDUS'].min()
min_deviation_CHAS = bostondf['CHAS'].min()
min_deviation_NOX = bostondf['NOX'].min()
min_deviation_RM = bostondf['RM'].min()
min_deviation_AGE = bostondf['AGE'].min()
min_deviation_DIS = bostondf['DIS'].min()
min_deviation_RAD = bostondf['RAD'].min()
min_deviation_TAX = bostondf['TAX'].min()
min_deviation_PTRATIO = bostondf['PTRATIO'].min()
min_deviation_LSTAT = bostondf['LSTAT'].min()
min_MEDV = bostondf['MEDV'].min()

max_deviation_CRIM = bostondf['CRIM'].max()
max_deviation_ZN = bostondf['ZN'].max()
max_deviation_INDUS = bostondf['INDUS'].max()
max_deviation_CHAS = bostondf['CHAS'].max()
max_deviation_NOX = bostondf['NOX'].max()
max_deviation_RM = bostondf['RM'].max()
max_deviation_AGE = bostondf['AGE'].max()
max_deviation_DIS = bostondf['DIS'].max()
max_deviation_RAD = bostondf['RAD'].max()
max_deviation_TAX = bostondf['TAX'].max()
max_deviation_PTRATIO = bostondf['PTRATIO'].max()
max_deviation_LSTAT = bostondf['LSTAT'].max()
max_MEDV = bostondf['MEDV'].max()


std_deviation_CRIM = bostondf['CRIM'].std()
std_deviation_ZN = bostondf['ZN'].std()
std_deviation_INDUS = bostondf['INDUS'].std()
std_deviation_CHAS = bostondf['CHAS'].std()
std_deviation_NOX = bostondf['NOX'].std()
std_deviation_RM = bostondf['RM'].std()
std_deviation_AGE = bostondf['AGE'].std()
std_deviation_DIS = bostondf['DIS'].std()
std_deviation_RAD = bostondf['RAD'].std()
std_deviation_TAX = bostondf['TAX'].std()
std_deviation_PTRATIO = bostondf['PTRATIO'].std()
std_deviation_LSTAT = bostondf['LSTAT'].std()
std_deviation_MEDV = bostondf['MEDV'].std()
