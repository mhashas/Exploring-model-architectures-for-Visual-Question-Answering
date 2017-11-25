import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Merge, Dropout, Input
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

    dictionary = dict()
    top_words = None
    embedding_vector_length = None

    visual_model = None

    def prepareData(self,file, question_max_length = question_maxlen, answer_max_length = answer_maxlen):
        with open(file) as csv_file:
            train_data = csv.reader(csv_file, delimiter=self.dictionary.pp_data.csv_delimiter)
            X = []
            X_features = []
            Y = []

            for row in train_data:
                image_id = row[0]
                visual_features = self.dictionary.pp_data.img_features[self.dictionary.pp_data.image_to_visual_feat_mapping[image_id]]
                X_features.append(visual_features)

                question = row[1]
                question = question.lower().strip().strip('?!.').split()
                question_length = len(question)

                complete_answer = row[2]
                x_question = np.zeros(question_max_length)
                y = np.zeros(answer_max_length)

                try:
                    for i in range(question_max_length):
                        if i < question_length:
                            word = question[i]
                            x_question[i] = self.dictionary.word2idx.get(word, self.dictionary.word2idx[self.dictionary.oov])
                        else:
                            break
                    y = self.dictionary.labels2idx.get(complete_answer, self.dictionary.labels2idx[self.dictionary.oov])
                    Y.append(y)
                    #x_final = [visual_features, x_question]
                    X.append(x_question)
                except Exception as e:
                    print(str(e))
                    pass


            X_return = np.array(X)
            X_features = np.array(X_features)
            Y_return = np_utils.to_categorical(Y)
            return (X_return, X_features, Y_return)


    def buildVisualModel(self, X, Y):
        image_model = Sequential()
        image_dimension = self.dictionary.pp_data.calculateImageDimension()
        image_model.add(Reshape((image_dimension,), input_shape = (image_dimension,)))

        language_model = Sequential()
        language_model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        language_model.add(keras_lstm(512, dropout=0.2, recurrent_dropout=0.2))

        model = Sequential()
        model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))

        """
        TODO RADU: dense layers ??
        TODO RADU: dropout_layers
        """

        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model

    ## TODO RADU: dropout ?
    def buildLanguageModel(self, X, Y):
        model = Sequential()
        model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        model.add(keras_lstm(512, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(Y.shape[1], activation='softmax')) # TODO RADU : use maxclasses instead of shape ?
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
        X, X_features, Y = self.prepareData(data_folder + train_data_file, self.question_maxlen)

        model = self.getModel(X, Y)
        model.fit([X, X_features], Y, epochs=10, batch_size=64)

        if save:
            self.saveModel(model, save_name)

        if return_model:
            return model

    def saveModel(self, model, model_name = 'test-saving-model.hdf5'):
        model.save(model_folder + model_name, overwrite=True)


    def evaluate(self, model, test_data_file = test_data_write_file):
        X, X_features, Y = self.prepareData(data_folder + test_data_file, self.question_maxlen)
        scores, acc = model.evaluate([X, X_features], Y)
        print(scores)
        print(acc)