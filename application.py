from preprocess import Preprocess
from dictionary import Dictionary
from lstm import LSTM
from keras.models import load_model
from constants import *
from bow import BOW

if __name__ == "__main__":
    # TODO: add args
    model_exists = 0

    helper = Preprocess()
    helper.preprocess()

    dictionary = Dictionary(helper)
    lstm = LSTM(dictionary, visual_model=False)
    bow = BOW(dictionary, visual_model=True)

    if (not model_exists):
        model = lstm.train(save=True, save_name = 'refactoring_stuff_bow.hdf5')
    else:
        model = load_model(model_folder + 'test-saving-model-image-features.hdf5')
    bow.evaluate(model)

