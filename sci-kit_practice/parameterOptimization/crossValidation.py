import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold


def main():
    dataset = pd.read_csv('../data/felicidad.csv')
    x = dataset.drop(['country', 'score'], 'columns')
    y = dataset['score']

    model = DecisionTreeRegressor()

    # cross_val_score
    score = cross_val_score(model, x, y, cv=10, scoring='neg_mean_squared_error')
    print('Score:', np.abs(np.mean(score)))

    # KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train, test)


if __name__ == "__main__":
    main()
