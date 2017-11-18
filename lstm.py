import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM as keras_lstm
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import json
import csv
from Dictionary import Dictionary
from keras.utils import np_utils
from constants import *
import sys

class LSTM:
    dictionary = dict()
    question_maxlen = None
    top_words = None
    embedding_vector_length = None

    def prepareData(self,file, question_max_length):
        with open(file) as csv_file:
            train_data = csv.reader(csv_file, delimiter=self.dictionary.pp_data.csv_delimiter)
            X = []
            Y = []

            for row in train_data:
                question = row[1]
                question = question.lower().strip().strip('?!.').split()
                question_length = len(question)

                answer = row[2]
                x = np.zeros(question_max_length)

                try:
                    for i in range(question_max_length):
                        if i < question_length:
                            x[i] = self.dictionary.word2idx[question[i]]
                        else:
                            break
                    Y.append(self.dictionary.labels2idx[answer])
                    X.append(x)
                except Exception as e:
                    pass

            print(len(X))
            print(len(Y))

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



    def train(self, train_data_file = train_data_write_file, save = False):
        X_train, Y_train = self.prepareData(data_folder + train_data_file, self.question_maxlen)

        model = self.buildModel(X_train, Y_train)
        model.fit(X_train, Y_train, epochs=10, batch_size=64)

        if save:
            self.saveModel(model)

    def saveModel(self, model):
        model_fn = 'test-saving-model.hdf5'
        model.save(data_folder + model_fn, overwrite=True)



    def evaluate(self):
        return
