from constants import *
import csv
import numpy as np
from keras.utils import np_utils
from dictionary import Dictionary

def prepare_data(file, dictionary : Dictionary, question_max_length=30):
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=dictionary.pp_data.csv_delimiter)

        X = []
        X_img_features = []
        Y = []

        for row in data:
            image_id = row[0]
            visual_features = dictionary.pp_data.img_features[dictionary.pp_data.image_to_visual_feat_mapping[image_id]]
            X_img_features.append(visual_features)

            question = row[1]
            question = question.lower().strip().strip('?!.').split()
            question_length = len(question)

            complete_answer = row[2]
            x_question = np.zeros(question_max_length)

            try:
                for i in range(question_max_length):
                    if i < question_length:
                        word = question[i]
                        x_question[i] = dictionary.word2idx.get(word, dictionary.word2idx[dictionary.oov])
                    else:
                        break
                y = dictionary.labels2idx.get(complete_answer, dictionary.labels2idx[dictionary.oov])

                Y.append(y)
                X.append(x_question)
            except Exception as e:
                print(str(e))
                pass

        X_return = np.array(X)
        X_img_features = np.array(X_img_features)
        Y_return = np_utils.to_categorical(Y)


        return (X_return, X_img_features, Y_return)


def analyse_results():
    return