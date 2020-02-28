import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df_hearth = pd.read_csv('./data/heart.csv')
    df_features = df_hearth.drop(['target'], 'columns')
    df_target = df_hearth['target']

    df_features = StandardScaler().fit_transform(df_features)

    x_train, x_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    pca = PCA(n_components=3)
    pca.fit(x_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    plt.plot(pca.explained_variance_ratio_)
    # plt.show()

    logistic = LogisticRegression()
    df_train = pca.transform(x_train)
    df_test = pca.transform(x_test)
    logistic.fit(df_train, y_train)
    print('Score PCA: ', logistic.score(df_test, y_test)) # 0.78

    logistic2 = LogisticRegression()
    df_train = ipca.transform(x_train)
    df_test = ipca.transform(x_test)
    logistic2.fit(df_train, y_train)
    print('Score IPCA: ', logistic2.score(df_test, y_test)) # 0.80
