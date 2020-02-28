import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    df_heart = pd.read_csv('../data/heart.csv')
    x = df_heart.drop(['target'], 'columns')
    y = df_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    knn_class = KNeighborsClassifier()
    knn_class.fit(x_train, y_train)
    knn_pred = knn_class.predict(x_test)
    print('KNN accuracy:', accuracy_score(y_test, knn_pred))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50)
    bag_class.fit(x_train, y_train)
    bag_pred = bag_class.predict(x_test)
    print('Bagging accuracy:', accuracy_score(y_test, bag_pred))


if __name__ == "__main__":
    main()
