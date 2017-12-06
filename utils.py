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
        X_question_id = []

        for row in data:
            question_id = row[0]
            image_id = row[1]

            X_question_id.append(question_id)

            visual_features = dictionary.pp_data.img_features[dictionary.pp_data.image_to_visual_feat_mapping[image_id]]
            X_img_features.append(visual_features)

            question = row[2]
            question = question.lower().strip().strip('?!.').split()
            question_length = len(question)

            complete_answer = row[3]
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


        return (X_return, X_img_features, Y_return, Y, X_question_id)


def analyse_results(inputs, predictions, answers, question_ids, model : Sequential, dictionary : Dictionary, accuracy, model_name):
    results = build_list_of_qpa_dictionaries(inputs, predictions, answers, question_ids, dictionary, model_name)
    statistics = get_statistics(results, dictionary)




def get_statistics(results, dictionary):
    number_of_results = len(results)

    number_of_correct_results = dict()
    statistics = dict()

    for result in results:
        print(result)
        exit(1)
    return


# returns list of dictionaries. Dictionary format is ['img_id', 'question', 'prediction', 'answer', 'correct']
def build_list_of_qpa_dictionaries(inputs, predictions, answers, question_ids, dictionary : Dictionary, model_name):
    N = len(predictions)
    results = list()

    test_data = dictionary.pp_data.test_data

    for i in range(N):
        predictions_for_question = predictions[i]
        prediction_idx_for_question = np.argmax(predictions_for_question)

        question_id = question_ids[i]
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
        question_info = test_data[question_id]['annotations']

        result = dict()

        result['image_id'] = question_info['image_id']
        result['question_id'] = question_id
        result['question_name'] = question
        result['question_type'] = question_info['question_type']
        result['question_multiple_choice'] = question_info['multiple_choice_answer']
        result['answer_type'] = question_info['answer_type']
        result['prediction'] = dictionary.idx2labels[int(prediction_idx_for_question)]
        result['answer'] = dictionary.idx2labels[int(answer)]
        result['top5'] = [dictionary.idx2labels[int(prediction)] for prediction in top5predictions]

        results.append(result)

    np.save(data_folder + results_write_file + model_name, results)
    return results
