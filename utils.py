from constants import *
import csv
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from dictionary import Dictionary

def prepare_data(file, dictionary : Dictionary, question_max_length=30):
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=dictionary.pp_data.csv_delimiter)

        X = []
        X_img_features = []
        Y = []
        X_img_ids = []

        for row in data:
            image_id = row[0]
            X_img_ids.append(image_id)

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


        return (X_return, X_img_features, Y_return, Y, X_img_ids)


def analyse_results(inputs, predictions, answers, image_ids, model : Sequential, dictionary : Dictionary, accuracy, model_name):
    results = build_list_of_qpa_dictionaries(inputs, predictions, answers, image_ids, dictionary)
    statistics = get_statistics(results, dictionary)




def get_statistics(results, dictionary):
    number_of_results = len(results)

    number_of_correct_results = dict()
    statistics = dict()
    return


# returns list of dictionaries. Dictionary format is ['img_id', 'question', 'prediction', 'answer', 'correct']
def build_list_of_qpa_dictionaries(inputs, predictions, answers, image_ids, dictionary : Dictionary):
    N = len(predictions)
    results = list()

    for i in range(N):
        predictions_for_question = predictions[i]
        prediction_idx_for_question = np.argmax(predictions_for_question)


        answer = int(answers[i])

        question_embed = inputs[i]
        question = ''

        for j in range(10):
            idx = question_embed[j]
            word = dictionary.idx2word[int(idx)]

            question += word

            if j < 9 and not question_embed[j + 1] == 0:
                question += ' '
            else:
                break

        top5predictions = predictions_for_question.argsort()[-5:][::-1]

        result = dict()
        result['image_id'] = image_ids[i]
        result['question'] = question
        result['prediction'] = dictionary.idx2labels[int(prediction_idx_for_question)]
        result['answer'] = dictionary.idx2labels[int(answer)]
        result['top1'] = int(answer) == int(prediction_idx_for_question)
        result['top5'] = 1 if answer in top5predictions else 0

        results.append(result)

    return results
