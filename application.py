from preprocess import Preprocess
from dictionary import Dictionary
from lstm import LSTM
from keras.models import load_model
from constants import *
from bow import BOW
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_question_len" , help="maximum of words of a question", default=30, type=int)
    parser.add_argument("--embedding_length", help="embedding layer output dimension", default=300, type=int)
    parser.add_argument("--max_answers", help="max labels stored for c", default=1000, type=int)
    parser.add_argument("--nr_epochs", help="number of epochs used for training", default=10, type=int)
    parser.add_argument("--batch_size", help="batch size used for training", default=64, type=int)
    parser.add_argument("--model_name", help="name of model to load in /models", default='', type=str)
    parser.add_argument("--lstm_hidden_units", help="number of hidden units in lstm", default=512, type=int)
    parser.add_argument("--deep_lstms", help="if we should use a deep lstm architecture", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--r_dropout", type=float, default=0.2)
    parser.add_argument("--visual_model", type=bool, default=True)

    args = parser.parse_args()

    helper = Preprocess()
    helper.preprocess()

    dictionary = Dictionary(helper, args.max_answers)
    lstm = LSTM(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length, visual_model=args.visual_model, lstm_hidden_units=args.lstm_hidden_units, dropout=args.dropout, recurrent_dropout=args.r_dropout, deeper_lstm=args.deep_lstms)
    bow = BOW(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length, visual_model=args.visual_model)

    if (not args.model_name):
        model = bow.train(save=True, save_name='refactoring_stuff_bow.hdf5')
    else:
        model = load_model(model_folder + args['model_name'])
    bow.evaluate(model)

