import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df_hearth = pd.read_csv('../data/heart.csv')
    df_features = df_hearth.drop(['target'], 'columns')
    df_target = df_hearth['target']

    df_features = StandardScaler().fit_transform(df_features)

    x_train, x_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(x_train)

    df_train = kpca.transform(x_train)
    df_test = kpca.transform(x_test)
    logistic = LogisticRegression()
    logistic.fit(df_train, y_train)
    print('Score KPCA:', logistic.score(df_test, y_test))  # 0.79
