from constants import *
import json
import gzip
import os
import numpy as np
import h5py
from collections import Counter
import codecs

class Preprocess:
    q_data_train = None
    q_data_test = None
    a_data_train = None
    a_data_test = None

    train_data = dict()
    test_data = dict()
    image_to_visual_feat_mapping = None
    img_features = np.asarray(h5py.File(data_folder + h5_file)['img_features'])

    csv_delimiter = '~'

    reader = codecs.getreader("utf-8")


    def __init__(self):
        with open(data_folder + json_file, 'r') as f:
            self.image_to_visual_feat_mapping = json.load(f)['VQA_imgid2id']

        with gzip.GzipFile(data_folder + q_train_data_file, 'r') as file:
            self.q_data_train = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + a_train_data_file, 'r') as file:
            self.a_data_train = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + q_test_data_file, 'r') as file:
            self.q_data_test = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + a_test_data_file, 'r') as file:
            self.a_data_test = json.load(self.reader(file))


    def getFeatures(self, answers):
        # for now get most common answer
        return self.getMostCommonAnswer(answers)


    def getMostCommonAnswer(self, answers):
        most_common_answers = dict()
        for answer_dict in answers:
            if not answer_dict['answer'] in most_common_answers:
                most_common_answers[answer_dict['answer']] = 1
            else:
                most_common_answers[answer_dict['answer']] += 1

        return str(Counter(most_common_answers).most_common()[0][0])

    def preprocessData(self, q_data, a_data, write_file):
        save_variable = dict()
        writeFile = open(write_file, 'w')

        for index, question_info in enumerate(q_data['questions']):
            image_id = question_info['image_id']
            question = question_info['question']
            save_variable[image_id] = dict()
            save_variable[image_id]['question'] = question
            save_variable[image_id]['annotations'] = a_data['annotations'][index]
            save_variable[image_id]['image_features'] = self.img_features[self.image_to_visual_feat_mapping[str(image_id)]]

            features = self.getFeatures(a_data['annotations'][index]['answers'])
            training_example = str(image_id) + self.csv_delimiter + str(question) + self.csv_delimiter + features
            writeFile.write(training_example + '\n')

        writeFile.close()
        return save_variable

    def preprocess(self, ignore_ex_check = False):
        if ignore_ex_check and os.path.isfile(data_folder + train_data_write_file) and os.path.isfile(data_folder + test_data_write_file):
            return

        self.train_data = self.preprocessData(self.q_data_train, self.a_data_train, data_folder + train_data_write_file)
        self.test_data = self.preprocessData(self.q_data_test, self.a_data_test, data_folder + test_data_write_file)
