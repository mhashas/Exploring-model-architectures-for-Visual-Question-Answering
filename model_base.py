from constants import *
from utils import *
from dictionary import Dictionary
from keras.models import Sequential


class ModelBase:
    question_maxlen = None
    top_words = None # TODO RADU : rename to more appropriate
    embedding_vector_length = None

    dictionary = dict()
    visual_model = None

    def __init__(self, dictionary : Dictionary, question_maxlen = 20, embedding_vector_length = 300, visual_model=True):
        self.dictionary = dictionary
        self.question_maxlen = question_maxlen
        self.embedding_vector_length = embedding_vector_length
        self.visual_model = visual_model
        self.top_words = len(self.dictionary.word2idx) + 1

    def build_visual_model(self, X, Y):
        raise NotImplementedError

    def build_language_model(self, X, Y):
        raise NotImplementedError

    # @re
    def get_model(self, X, Y) -> Sequential:
        if self.visual_model:
            return self.build_visual_model(X,Y)
        else:
            return self.build_language_model(X,Y)

    def train(self, train_data_file=train_data_write_file, return_model=True, save=False, save_name=''):
        X, X_features, Y = prepare_data(data_folder + train_data_file, self.dictionary, self.question_maxlen)

        model = self.get_model(X, Y)

        if self.visual_model:
            model.fit([X, X_features], Y, epochs=10, batch_size=64)
        else:
            model.fit(X, Y, epochs=10, batch_size=64)

        if save:
            self.get_model(model, save_name)

        if return_model:
            return model

    def evaluate(self, model : Sequential, test_data_file=test_data_write_file, analyse_results=True):
        X, X_features, Y = prepare_data(data_folder + test_data_file, self.dictionary, self.question_maxlen)

        if self.visual_model:
            scores, acc = model.evaluate([X, X_features], Y)
        else:
            scores, acc = model.evaluate(X, Y)

        if analyse_results:
            predictions = None
            if self.visual_model:
                predictions = model.predict([X, X_features])
            else:
                predictions = model.predict(X)
            # START ANALISING STUFF YO

    def save_model(self, model : Sequential, model_name):
        model.save(model_folder + model_name, overwrite=True)