from keras.models import Sequential
from keras.layers import Dense, Reshape, Merge, Dropout, Input
from keras.layers import LSTM as keras_lstm
from keras.layers.embeddings import Embedding
from dictionary import Dictionary
from constants import *
from model_base import ModelBase

class LSTM(ModelBase):

    def __init__(self, dictionary, question_maxlen=20, embedding_vector_length=300, visual_model=True):
        super(LSTM, self).__init__(dictionary, question_maxlen, embedding_vector_length, visual_model)


    def build_visual_model(self, X, Y):
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

    def build_language_model(self, X, Y):
        model = Sequential()
        model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        model.add(keras_lstm(512, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model