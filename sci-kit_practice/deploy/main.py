from utils import Utils
from models import Model

if __name__ == "__main__":
    utils = Utils()
    model = Model()

    data = utils.load_from_csv('./in/felicidad.csv')
    x, y = utils.features_target(data, ['score', 'rank', 'country'], 'score')

    model.grid_training(x, y)
