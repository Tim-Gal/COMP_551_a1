import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]  # add a dimension for the features
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])  # add bias by adding a constant feature of value 1
        self.w = np.linalg.lstsq(x, y, rcond=None)[0]  # return w for the least square difference
        return self

    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        yh = x @ self.w  # predict the y values
        return yh


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


def calculate_stats():
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


def calculate_mse(true, pred):
    return np.mean((true - pred) ** 2)


if __name__ == '__main__':

    # ----- BOSTON ------
    bostondf = pd.read_csv("boston.csv")
    bostondf_dropped = bostondf.dropna(axis=0, how='any', inplace=False)
    bostondf_cleaned = remove_outliers(bostondf_dropped)

    features = bostondf_cleaned[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']].values
    labels = bostondf_cleaned[['MEDV']].values

    # ----- HOLDOUT VALIDATION -----
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    train_mse = calculate_mse(Y_train, Y_train_pred)
    test_mse = calculate_mse(Y_test, Y_test_pred)

    print(f"model performance (MSE) on training set ({test_size*100}% of data set): {train_mse}")
    print(f"model performance (MSE) on testing set ({100-test_size*100} of data set): {test_mse}")

    # ----- CROSS VALIDATION -----
    k = 5
    fold_size = len(features) // k
    train_mse_values = []
    test_mse_values = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else None

        X_test_fold = features[start:end]
        Y_test_fold = labels[start:end]
        X_train_fold = np.vstack((features[:start], features[end:])) if end is not None else features[:start]
        Y_train_fold = np.vstack((labels[:start], labels[end:])) if end is not None else labels[:start]

        model = LinearRegression()
        model.fit(X_train_fold, Y_train_fold)

        Y_train_pred = model.predict(X_train_fold)
        Y_test_pred = model.predict(X_test_fold)

        train_mse_fold = calculate_mse(Y_train_fold, Y_train_pred)
        test_mse_fold = calculate_mse(Y_test_fold, Y_test_pred)

        train_mse_values.append(train_mse_fold)
        test_mse_values.append(test_mse_fold)

    train_avg_mse = np.mean(train_mse_values)
    test_avg_mse = np.mean(test_mse_values)
    print("Average model performance (MSE) across 5 folds, on training set:", train_avg_mse)
    print("Average model performance (MSE) across 5 folds, on testing set:", test_avg_mse)

    # ----- LEARNING CURVE ANALYSIS -----
    train_mse_values = []
    test_mse_values = []
    training_sizes = []

    for test_size in range(2, 9):
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=test_size/10, random_state=42)
        training_sizes.append(test_size*10)

        model = LinearRegression()
        model.fit(X_train, Y_train)

        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)

        train_mse = calculate_mse(Y_train, Y_train_pred)
        test_mse = calculate_mse(Y_test, Y_test_pred)

        train_mse_values.append(train_mse)
        test_mse_values.append(test_mse)

        print(f"model performance (MSE) on training set ({test_size * 10}% of data set): {train_mse}")
        print(f"model performance (MSE) on testing set ({100 - test_size * 10}% of data set): {test_mse}")

    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, train_mse_values, label="Training MSE", marker='o')
    plt.plot(training_sizes, test_mse_values, label="Test MSE", marker='o')
    plt.xlabel("Training Size (%)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- WINE ------
    winedf = pd.read_csv("wine.data")
    winedf_dropped = winedf.dropna(axis=0, how='any', inplace=False)
    winedf_cleaned = remove_outliers(winedf_dropped)
