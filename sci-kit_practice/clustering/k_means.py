import pandas as pd

from sklearn.cluster import MiniBatchKMeans
# Kmeans funciona mejor para cuando el número de centroides ya está definido


def main():
    dataset = pd.read_csv('../data/candy.csv')

    x = dataset.drop('competitorname', 'columns')

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)

    print('Number of clusters:', len(kmeans.cluster_centers_))
    print('-' * 64)
    print(kmeans.predict(x))

    dataset['group'] = kmeans.predict(x)


if __name__ == "__main__":
    main()
