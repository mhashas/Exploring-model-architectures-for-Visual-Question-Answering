from preprocess import Preprocess
from dictionary import Dictionary
from lstm import LSTM
from keras.models import load_model
from constants import *

if __name__ == "__main__":
    # TODO: add args
    model_exists = 0

    helper = Preprocess()
    helper.preprocess()

    dictionary = Dictionary(helper)
    lstm = LSTM(dictionary, visual_model=True)

    if (not model_exists):
        model = lstm.train(save=True, save_name = 'refactoring_stuff.hdf5')
    else:
        model = load_model(model_folder + 'test-saving-model-image-features.hdf5')
    lstm.evaluate(model)

