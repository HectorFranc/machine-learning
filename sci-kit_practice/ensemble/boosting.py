import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    df_heart = pd.read_csv('../data/heart.csv')
    x = df_heart.drop(['target'], 'columns')
    y = df_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    boost = GradientBoostingClassifier(n_estimators=50).fit(x_train, y_train)
    boost_pred = boost.predict(x_test)
    print('-' * 64)
    print('Gradient boosting accuracy:', accuracy_score(y_test, boost_pred))


if __name__ == "__main__":
    main()
