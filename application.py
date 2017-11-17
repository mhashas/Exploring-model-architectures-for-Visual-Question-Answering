from Preprocess import Preprocess
from Dictionary import Dictionary
from lstm import LSTM

if __name__ == "__main__":
    preprocess = Preprocess()
    preprocess.preprocess()
    dictionary = Dictionary(preprocess)
    lstm = LSTM(dictionary)
    lstm.train(save=True)
