import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


def main():
    dataset = pd.read_csv('../data/felicidad.csv')
    x = dataset.drop(['country', 'rank', 'score'], 'columns')
    y = dataset['score']

    reg = RandomForestRegressor()

    parameters = {
        'n_estimators': range(4, 16),
        'criterion': ['mse', 'mae'],
        'max_depth': range(2, 10),
    }
    ran_est = RandomizedSearchCV(reg, parameters, n_iter=10, cv=3, scoring='neg_mean_absolute_error')
    ran_est.fit(x, y)

    print(ran_est.best_estimator_)
    print(ran_est.best_params_)


if __name__ == "__main__":
    main()
