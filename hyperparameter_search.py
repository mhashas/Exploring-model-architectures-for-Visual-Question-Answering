import argparse
from bow import BOW
from lstm import LSTM
from preprocess import Preprocess
from dictionary import Dictionary

def bow_hyperparameter_search(embedding_dims, max_question_lens, max_answers):
    helper = Preprocess()
    helper.preprocess()

    for max_answer in max_answers:
        dictionary = Dictionary(helper, max_answer)

        for embedding_dim in embedding_dims:
            for question_maxlen in max_question_lens:
                print('Building model BOW' + '-embedding_dim=' + str(embedding_dim) + '-question_maxlen=' + str(question_maxlen) + '-max_answers=' + str(max_answer))
                bow = BOW(dictionary, question_maxlen, embedding_dim, visual_model=True)
                model = bow.train(verbose=2)
                acc = bow.evaluate(model)
                print('Model evaluated, acc=' + str(acc))

def lstm_hyperparameter_search(embedding_dim, max_question_len, max_answers, number_hidden_units, dropouts, r_dropouts, number_stacked_lstms):
    helper = Preprocess()
    helper.preprocess()

    for max_answer in max_answers:
        dictionary = Dictionary(helper, max_answer)

        for number_h_units in number_hidden_units:
            for dropout in dropouts:
                for r_dropout in r_dropouts:
                    for number_stacked_lstm in number_stacked_lstms:
                        lstm = LSTM(dictionary, lstm_hidden_units=number_h_units, dropout=dropout, recurrent_dropout=r_dropout, number_stacked_lstms=number_stacked_lstm)


def training_hyperparameter_search(nr_epoch, batch_size):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter_type_search", type=str, default="bow")

    args = parser.parse_args()

    max_question_len = [10, 15, 20, 30]
    embedding_dim = [200, 300, 400 , 600]
    max_answers = [500, 1000, 2000, 4000]

    number_hidden_units = [256, 512, 768, 104]
    dropout = [0.2, 0.3, 0.4, 0.5]
    r_dropout = [0.2, 0.3, 0.4, 0.5]
    number_stacked_lstms = [0, 1, 2 , 3]

    nr_epoch = [5, 8, 10]
    batch_size = [32, 64, 128]

    if args.parameter_type_search == 'bow':
        bow_hyperparameter_search(embedding_dim, max_question_len, max_answers)
    elif args.parameter_type_search == 'lstm':
        lstm_hyperparameter_search(embedding_dim, max_question_len, max_answers, number_hidden_units, dropout, r_dropout, number_stacked_lstms)
    elif args.parameter_type_search == 'train':
        training_hyperparameter_search(nr_epoch, batch_size)