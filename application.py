from Preprocess import Preprocess
from Dictionary import Dictionary

if __name__ == "__main__":
    preprocess = Preprocess()
    preprocess.preprocess()
    dictionary = Dictionary(preprocess)
