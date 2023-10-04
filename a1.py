import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]  # add a dimension for the features
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])  # add bias by adding a constant feature of value 1
        self.w = np.linalg.lstsq(x, y, rcond=None)[0]  # return w for the least square difference
        return self

    def predict(self, x):
        print("x.shape : ", x.shape)
        print(x)
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])  # Add a column of ones for bias
        print("x.shape : ", x.shape)
        print(x)
        print("w.shape : ", self.w.shape)
        print(self.w)
        yh = x @ self.w  # Predict the y values
        return yh


class LogisticRegression:

    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # to get the tolerance for the norm of gradients
        self.max_iters = max_iters  # maximum number of iteration of gradient descent
        self.verbose = verbose

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        N, D = x.shape
        self.w = np.zeros(D)
        g = np.inf
        t = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            g = self.gradient(x, y)
            self.w = self.w - self.learning_rate * g
            t += 1

        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh

    def gradient(self, x, y):
        N, D = x.shape
        yh = logistic(np.dot(x, self.w))  # predictions  size N
        grad = np.dot(x.T, yh - y) / N  # divide by N because cost is mean over N points
        return grad  # size D


logistic = lambda z: 1./ (1 + np.exp(-z))       #logistic function


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


if __name__ == '__main__':

    bostondf = pd.read_csv("boston.csv")
    bostondf_dropped = bostondf.dropna(axis=0, how='any', inplace=False)
    bostondf_cleaned = remove_outliers(bostondf_dropped)

    winedf = pd.read_csv("wine.data")
    winedf_dropped = winedf.dropna(axis=0, how='any', inplace=False)
    winedf_cleaned = remove_outliers(winedf_dropped)

    features = bostondf_cleaned[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']].values
    labels = bostondf_cleaned[['MEDV']].values

    new_data = np.array([0.14476, 0, 10.01, 0, 0.547, 5.731, 65.2, 2.7592, 6, 432])

    LinearRegression_model = LinearRegression()
    LinearRegression_model.fit(features, labels)
    LinearRegression_predictions = LinearRegression_model.predict(new_data)
    print(LinearRegression_predictions)

    LogisticRegression_model = LogisticRegression()
    LogisticRegression_model.fit(features, labels)



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
