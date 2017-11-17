from constants import *
from Preprocess import Preprocess
import os
import re
import csv
import pickle
from collections import defaultdict
import numpy as np

class Dictionary:
    word2idx = {}
    idx2word = []
    labels2idx = {}
    labels = []
    pp_data = None # type: Preprocess
    max_classes = 1000

    def __init__(self, preprocessed_data, max_classes = 1000):
        self.pp_data = preprocessed_data # type: Preprocess
        self.max_classes = max_classes

        if os.path.isfile(data_folder + idx2word_file) and os.path.isfile(data_folder + word2idx_file):
            self.loadDictionaries()
        else:
            self.generateDictionaries(data_folder + train_data_write_file, True)
        self.topAnswers(data_folder + train_data_write_file)

    def generateDictionaries(self, processed_csv, write_out):
        with open(processed_csv, 'r') as csv_data:
            data = csv.reader(csv_data, delimiter=self.pp_data.csv_delimiter)
            for (_, question, answer) in data:
                words = question + ' ' + answer
                for word in re.split(r'[^\w]+', words):
                    lowercase_word = word.lower()
                    if lowercase_word not in self.word2idx:
                        index = len(self.idx2word)
                        self.idx2word.append(lowercase_word)
                        self.word2idx[lowercase_word] = index

        csv_data.close()

        if write_out:
            fd = open(data_folder + idx2word_file, 'wb')
            pickle.dump(self.idx2word, fd)
            fd.close()

            fd = open(data_folder + word2idx_file, 'wb')
            pickle.dump(self.word2idx, fd)
            fd.close()

    def loadDictionaries(self):
        fd = open(data_folder + word2idx_file, 'rb')
        self.word2idx = pickle.load(fd)
        fd.close()

        fd = open(data_folder + idx2word_file, 'rb')
        self.idx2word = pickle.load(fd)

    def getVocabSize(self):
        return len(self.idx2word)

    def getIdx(self, word):
        word = word.lower()
        if (word in self.word2idx):
            return self.word2idx[word]
        else:
            return -1

    def getBOW(self, str):
        bow = np.zeros(self.getVocabSize())

        words = re.split(r'[^\w]+', str)

        for word in words:
            idx = self.getIdx(word)
            if idx > -1:
                bow[idx] += 1

        return bow

    def topAnswers(self, processed_data_file, max_classes=1000):
        answers = defaultdict(int)

        with open(processed_data_file, 'r') as csv_data:
            data = csv.reader(csv_data, delimiter=self.pp_data.csv_delimiter)
            for (_, _, answer) in data:
                answers[answer.lower()] += 1

        csv_data.close()

        sorted_answers = sorted(answers, key=answers.get, reverse=True)
        self.labels = sorted_answers[0:max_classes]

        for i in range(len(self.labels)):
            self.labels2idx[self.labels[i]] = i

        return self.labels2idx