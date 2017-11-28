from keras.models import Sequential
from keras.layers import Dense, Reshape, Merge, Dropout, Input
from keras.layers import LSTM as keras_lstm
from keras.layers.embeddings import Embedding
from dictionary import Dictionary
from constants import *
from utils import *

class LSTM:
    question_maxlen = None
    top_words = None
    embedding_vector_length = None

    dictionary = dict()
    visual_model = None

    def buildVisualModel(self, X, Y):
        image_model = Sequential()
        image_dimension = self.dictionary.pp_data.calculateImageDimension()
        image_model.add(Reshape((image_dimension,), input_shape = (image_dimension,)))

        language_model = Sequential()
        language_model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        language_model.add(keras_lstm(512, dropout=0.2, recurrent_dropout=0.2)) # discover these parameters?
        model = Sequential()
        model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))

        """
        TODO RADU: dense layers ?? 
        TODO RADU: dropout_layers ??
        """

        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model

    def buildLanguageModel(self, X, Y):
        model = Sequential()
        model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        model.add(keras_lstm(512, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model


    def getModel(self, X, Y):
        if self.visual_model:
            return self.buildVisualModel(X,Y)
        else:
            return self.buildLanguageModel(X,Y)

    def __init__(self, dictionary, question_maxlen = 20, embedding_vector_length = 300, visual_model = True):
        self.dictionary = dictionary # type: Dictionary
        self.question_maxlen = question_maxlen
        self.embedding_vector_length = embedding_vector_length
        self.top_words = len(self.dictionary.word2idx) + 1
        self.visual_model = visual_model


    def train(self, train_data_file = train_data_write_file, save = False, return_model = True, save_name = 'visual_model_whateva.hdf5'):
        X, X_features, Y = prepare_data(data_folder + train_data_file, self.dictionary, self.question_maxlen)

        model = self.getModel(X, Y)

        if self.visual_model:
            model.fit([X, X_features], Y, epochs=10, batch_size=64)
        else:
            model.fit(X, Y, epochs=10, batch_size=64)

        if save:
            self.saveModel(model, save_name)

        if return_model:
            return model

    def saveModel(self, model, model_name = 'test-saving-model.hdf5'):
        model.save(model_folder + model_name, overwrite=True)


    def evaluate(self, model, test_data_file = test_data_write_file):
        X, X_features, Y = prepare_data(data_folder + test_data_file, self.dictionary, self.question_maxlen)

        if self.visual_model:
            scores, acc = model.evaluate([X, X_features], Y)
        else:
            scores, acc = model.evaluate(X, Y)

        print(scores)
        print(acc)