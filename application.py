from preprocess import Preprocess
from dictionary import Dictionary
from lstm import LSTM
from keras.models import load_model
from constants import *
from utils import *
from bow import BOW
import argparse
import numpy as np


def train_and_evaluate(args):
    helper = Preprocess()
    helper.preprocess()

    dictionary = Dictionary(helper, args.max_answers)
    lstm = LSTM(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length,
                visual_model=args.visual_model, lstm_hidden_units=args.lstm_hidden_units, dropout=args.dropout,
                recurrent_dropout=args.r_dropout, deeper_lstm=args.deep_lstms)
    bow = BOW(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length,
              visual_model=args.visual_model)

    if (not args.model_name):
        model = bow.train(save=True)
    else:
        model = load_model(model_folder + args.model_name)
    bow.evaluate(model)

def visualize_errors(args):
    helper = Preprocess()
    helper.preprocess()

    dictionary = Dictionary(helper, args.max_answers, args.include_question_mark)

    inputs = np.load(data_folder + inputs_data_file + "_" + args.model_type + npy_save_type)
    answers = np.load(data_folder + answers_data_file + "_" + args.model_type + npy_save_type)
    predictions = np.load(data_folder + predictions_data_file + "_" + args.model_type + npy_save_type)
    image_ids = np.load(data_folder + question_ids_data_file + "_" + args.model_type + npy_save_type)

    analyse_results(inputs, predictions, answers, image_ids, None, dictionary, 0, args.model_type)


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
    parser.add_argument("--only_analyze", type=bool, default=True)
    parser.add_argument("--include_question_mark", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="bow")

    args = parser.parse_args()

    if args.only_analyze:
        visualize_errors(args)
    else:
        train_and_evaluate(args)
