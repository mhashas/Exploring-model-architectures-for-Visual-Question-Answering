import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM as keras_lstm
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import json
import csv
from dictionary import Dictionary
from keras.utils import np_utils
from constants import *
import sys

class LSTM:
    question_maxlen = None
    answer_maxlen = 3

    max_classes = 1000

    dictionary = dict()
    top_words = None
    embedding_vector_length = None

    def prepareData(self,file, question_max_length = question_maxlen, answer_max_length = answer_maxlen):
        with open(file) as csv_file:
            train_data = csv.reader(csv_file, delimiter=self.dictionary.pp_data.csv_delimiter)
            X = []
            Y = []

            for row in train_data:
                question = row[1]
                question = question.lower().strip().strip('?!.').split()
                question_length = len(question)

                complete_answer = row[2]
                x = np.zeros(question_max_length)
                y = np.zeros(answer_max_length)

                try:
                    for i in range(question_max_length):
                        if i < question_length:
                            word = question[i]
                            x[i] = self.dictionary.word2idx.get(word, self.dictionary.word2idx[self.dictionary.oov])
                        else:
                            break
                    y = self.dictionary.labels2idx.get(complete_answer, self.dictionary.labels2idx[self.dictionary.oov])
                    Y.append(y)
                    X.append(x)
                except Exception as e:
                    print(str(e))
                    pass


            X_return = np.array(X)
            Y_return = np_utils.to_categorical(Y)
            return (X_return, Y_return)


    def buildModel(self, X_train, Y_train):
        model = Sequential()
        model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        model.add(keras_lstm(512, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(Y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def __init__(self, dictionary, question_maxlen = 20, embedding_vector_length = 300):
        self.dictionary = dictionary # type: Dictionary
        self.question_maxlen = question_maxlen
        self.embedding_vector_length = embedding_vector_length
        self.top_words = len(self.dictionary.word2idx) + 1



    def train(self, train_data_file = train_data_write_file, save = False, return_model = True):
        X_train, Y_train = self.prepareData(data_folder + train_data_file, self.question_maxlen)

        model = self.buildModel(X_train, Y_train)
        model.fit(X_train, Y_train, epochs=10, batch_size=64)

        if save:
            self.saveModel(model, 'test-saving-model-1-epoch.hdf5')

        if return_model:
            return model

    def saveModel(self, model, model_name = 'test-saving-model.hdf5'):
        model.save(model_folder + model_name, overwrite=True)



    def evaluate(self, model, test_data_file = test_data_write_file):
        X_test, Y_test = self.prepareData(data_folder + test_data_file, self.question_maxlen)
        scores, acc = model.evaluate(X_test, Y_test)
        print(scores)
        print(acc)