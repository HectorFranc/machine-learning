import pandas as pd

from sklearn.cluster import MeanShift
# MeanShift funciona mejor para descubrir el número de centroides correcto para el clustering


def main():
    dataset = pd.read_csv('../data/candy.csv')
    x = dataset.drop('competitorname', 'columns')

    meanshift = MeanShift().fit(x)

    print('Labels:', meanshift.labels_)  # Etiquetas para el dataset entregado
    print('N centers:', max(meanshift.labels_) + 1)  # Numero de centroides encontrados
    print('Centers:', meanshift.cluster_centers_)  # Ubicación de los centroides

    dataset['meanshift'] = meanshift.labels_


if __name__ == "__main__":
    main()
