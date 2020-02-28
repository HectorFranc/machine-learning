import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('../data/felicidad.csv')
    x = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    modelLinear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = modelLinear.predict(x_test)

    modelLasso = Lasso().fit(x_train, y_train)
    y_predict_lasso = modelLasso.predict(x_test)

    modelRidge = Ridge().fit(x_train, y_train)
    y_predict_ridge = modelRidge.predict(x_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('Linear loss:', linear_loss)  # 7.29e-08
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('Lasso loss:', lasso_loss)  # 1.09
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print('Ridge loss:', ridge_loss)  # 0.0052

    print('=' * 32)
    print('Coeficients Lasso', modelLasso.coef_)
    print('Coeficients Ridge', modelRidge.coef_)
