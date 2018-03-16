# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import string
import os
import sys
import csv
import editdistance as ed
import numpy as np
import tensorflow as tf
from flask import Flask
from flask import jsonify

import seq2seq_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Hides verbose log messages
app = Flask(__name__, static_url_path="/static")
app.config['JSON_AS_ASCII'] = False


FLAGS = object()
_buckets = [(35, 35)]

best_wer = 1.0

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

gr_vocab = None
ph_vocab = None
rev_gr_vocab = None
rev_ph_vocab = None
dict_phoneme_property = {}
list_phoneme_property_weights = []

try:
    xrange
except NameError:
    xrange = range


def parse_cmd():
    """Parse command line options."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--train",
                        dest="train", action="store_true",
                        help="Evaluate performance from the best saved model")

    parser.add_argument("-nt", "--no-train",
                        dest="test", action="store_true",
                        help="Evaluate performance from the best saved model")

    parser.add_argument("-i", "--interactive",
                        dest="interactive", action="store_true",
                        help="Run model in interactive mode")

    parser.add_argument("-e", "--evaluate",
                        dest="evaluate", action="store_true",
                        help="Run model in interactive mode")

    parser.add_argument("-s", "--serve",
                        dest="serve", action="store_true",
                        help="Run model in server mode")
    parser.add_argument("-H", "--host", default="0.0.0.0",
                        help="Define the host for the server mode")
    parser.add_argument("-p", "--port", default=5000,
                        help="Define the port for the server mode")

    parser.add_argument("-f", "--file_loc",
                        default="./cmu_data/Test.txt", type=str,
                        help="Evaluation file location")

    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    parser.set_defaults(interactive=False)
    parser.set_defaults(evaluate=False)
    parser.set_defaults(serve=False)

    parser.add_argument("-cd", "--checkpoint_dir",
                        default="./model_checkpoints", type=str,
                        help="Training checkpoint directory")

    args = parser.parse_args()
    arg_dict = vars(args)
    return arg_dict

def initialize_vocabulary(vocabulary_path):
    """Load vocabulary from given file.

    Args:
        vocabulary_path: Path of vocabulary file.
    Returns:
        vocab: A dictionary mapping elements of vocabulary to index
        rev_vocab: A list mapping index to element of vocabulary
    """
    if os.path.isfile(vocabulary_path):
        reader = csv.reader(open(vocabulary_path, 'r'))
        vocab_table = {}
        for row in reader:
            k, v = row
            vocab_table[k] = v
        rev_vocab_table = {v: k for k, v in vocab_table.items()}
        return (vocab_table, rev_vocab_table)
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)
        return {}, []


def setModelParameters():
    """
    Reading model parameters from a json file

    :return parameters: a dictionary containing all the model parameters
    """
    global gr_vocab,ph_vocab,rev_gr_vocab,rev_ph_vocab

    with open('parameters.json', 'r') as fp:
        parameters = json.load(fp)

    gr_vocab, rev_gr_vocab = initialize_vocabulary(parameters['source_vocab_path'])
    ph_vocab, rev_ph_vocab = initialize_vocabulary(parameters['target_vocab_path'])



    parameters['source_vocab_size'] = len(gr_vocab)
    parameters['target_vocab_size'] = len(ph_vocab)

    return parameters


def model_graph(session, isTraining):
    """Create the model graph by creating an instance of Seq2SeqModel."""
    return seq2seq_model.Seq2SeqModel(
        _buckets, isTraining, FLAGS['max_gradient_norm'], FLAGS['batch_size'],
        FLAGS['learning_rate'], FLAGS['learning_rate_decay_factor'],
        FLAGS)


def get_model(session, isTraining):
    model = model_graph(session, isTraining=isTraining)
    mckpts = tf.train.get_checkpoint_state(FLAGS['model_checkpoint_path'])

    steps_done = 0
    try:
        # Restore model parameters
        print(mckpts.model_checkpoint_path)
        model.saver.restore(session, mckpts.model_checkpoint_path)
        sys.stdout.write("Loading model parameters from %s\n" % mckpts.model_checkpoint_path)
        sys.stdout.flush()
        steps_done = int(mckpts.model_checkpoint_path.split('-')[-1])
        print("loaded from %d completed steps" % (steps_done))
    except:
        sys.stdout.write("Creating a fresh model with defined parameters.\n")
        sys.stdout.flush()
        # Initialize model parameters
        session.run(tf.global_variables_initializer())
    return model, steps_done


def calculate_phoneme_similarity(original_ph_seq, predicted_ph_seq):
    global pad
    global dict_phoneme_property, list_phoneme_property_weights

    length = 0
    pad_length = 0

    original_ph_seq = original_ph_seq.replace("ˈ", "")
    original_ph_seq = original_ph_seq.replace("ˌ", "")
    original_ph_seq = original_ph_seq.replace("-", "")

    predicted_ph_seq = predicted_ph_seq.replace("ˈ", "")
    predicted_ph_seq = predicted_ph_seq.replace("ˌ", "")
    predicted_ph_seq = predicted_ph_seq.replace("-", "")

    original_ph_seq_list = original_ph_seq.split()
    predicted_ph_seq_list = predicted_ph_seq.split()

    if len(original_ph_seq_list) >= len(predicted_ph_seq_list):
        length = len(original_ph_seq_list)
        pad_length = len(original_ph_seq_list) - len(predicted_ph_seq_list)
        for item in range(pad_length):
            predicted_ph_seq_list.append('pad')
    else:
        length = len(predicted_ph_seq_list)
        pad_length = len(predicted_ph_seq_list) - len(original_ph_seq_list)
        for item in range(pad_length):
            original_ph_seq_list.append('pad')

    total_difference = 0

    for index in range(length):
        original_seq = [bool(int(x)) for x in dict_phoneme_property[original_ph_seq_list[index]]]
        predicted_seq = [bool(int(x)) for x in dict_phoneme_property[predicted_ph_seq_list[index]]]
        difference_seq = np.asarray(np.logical_xor(original_seq, predicted_seq).astype(int).tolist())
        temp_weights = np.asarray([int(i) for i in list_phoneme_property_weights])
        weighted_seq = np.multiply(difference_seq, temp_weights)
        difference_score = np.sum(weighted_seq)
        total_difference += difference_score
    return total_difference


def calc_error(dictionary):
    """Calculate a number of prediction errors.
    """
    word_error = 0
    word_error_without_stress = 0
    phoneme_error = 0
    phoneme_error_without_stress = 0
    total_phoneme_similarity_error = 0.0
    total_phonemes = 0
    total_words = 0
    with tf.Session() as sess:
        # Load model
        with tf.variable_scope("model"):
            model, _ = get_model(sess, training=False)

            f = open("g2p_test_output.txt", "w")
            for word, pronunciations in dictionary.items():
                hyp = decode_word(sess, model, word)
                actual = "".join(pronunciations)
                total_words += 1
                phoneme_sim_error = calculate_phoneme_similarity(hyp, actual)
                distance_1 = ed.eval(hyp.split(), actual.split())
                pronunciations_stripped = ''.join(filter(lambda x: not x.isdigit(), actual))
                pronunciations_stripped = pronunciations_stripped.replace("ˈ", "")
                pronunciations_stripped = pronunciations_stripped.replace("ˌ", "")
                hyp_stripped = ''.join(filter(lambda x: not x.isdigit(), hyp))
                hyp_stripped = hyp_stripped.replace("ˈ", "")
                hyp_stripped = hyp_stripped.replace("ˌ", "")

                distance_2 = ed.eval(hyp_stripped.split(), pronunciations_stripped.split())

                f.write(
                    "[" + word + "] | [" + actual + "] | [" + hyp + "] | [" + hyp_stripped + "]  -- phoneme_sim_error - " + str(
                        phoneme_sim_error) + "\n")
                if distance_1 != 0:
                    word_error += 1

                if distance_2 != 0:
                    word_error_without_stress += 1
                    phoneme_error += distance_2
                    total_phonemes += len(hyp_stripped)

                total_phoneme_similarity_error += phoneme_sim_error

            f.write("\n\nTotal numbers of incorrect phonemes is: " + str(phoneme_error) + " out of : " + str(
                total_phonemes) + " possible phonemes.")
            f.close()
            try:
                word_error = float(word_error) / float(total_words)
                word_error_without_stress = float(word_error_without_stress) / float(total_words)
            except ZeroDivisionError:
                print("0 words in evaluation set")
                word_error = 1.0
            try:
                phoneme_error = float(phoneme_error) / float(total_phonemes)
            except ZeroDivisionError:
                print("0 phones in evaluation set")
                phoneme_error = 1.0

    return word_error, phoneme_error, word_error_without_stress


def decode_word(sess, model, word):
    """Decode input word to sequence of phonemes.
    Args:
      word: input word;
    Returns:
      phonemes: decoded phoneme sequence for input word;
    """
    global gr_vocab,ph_vocab,rev_gh_vocab,rev_ph_vocab

    bucket_id = 0
    # Get token-ids for the input word.
    token_ids = [gr_vocab.get(s, UNK_ID) for s in word]

    encoder_inputs, seq_len, decoder_inputs, seq_len_target = \
        model.get_batch([(token_ids, [])], bucket_id=bucket_id)
    # Get output logits for the word by running the model to get output_logits of shape TxBx|V|
    output_logits = model.step(sess, encoder_inputs, seq_len,
                               decoder_inputs, seq_len_target)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = np.argmax(output_logits, axis=1)
    # Reshape the output and make it batch major via transpose
    outputs = np.reshape(outputs, (max(seq_len_target), 1)).T
    cur_output = outputs[0].tolist()
    if EOS_ID in cur_output:
        cur_output = cur_output[:cur_output.index(EOS_ID)]
    return " ".join([rev_ph_vocab[str(output)] for output in cur_output])


def interactive():
    """Decode word from standard input.
    """
    with tf.Session() as sess:
        # Load model
        with tf.variable_scope("model"):
            model, _ = get_model(sess, isTraining=False)
        while True:
            word = input("> ")
            print(word)
            wordlist = word.split()
            output = ""
            for word in wordlist:
                punc_regex = str.maketrans('', '', string.punctuation)
                stripped_word = word.translate(punc_regex)
                output += decode_word(sess, model, stripped_word)
            print(output)


def evaluate():
    """Calculate and print out word error rate (WER) and Accuracy
           on test sample.

        Args:
          test_lines: List of test dictionary. Each element of list must be String
                    containing word and its pronounciation (e.g., "word W ER D");
        """
    test_dic = create_Testdict(FLAGS['file_loc'])

    if len(test_dic) < 1:
        print("Test dictionary is empty")
        return

    wer, per, wer_s = calc_error(test_dic)
    print('Beginning calculation word error rate (WER) and phoneme error rate (PER) on test sample of size: %d.' % (
        len(test_dic)))

    print('Total WER %.5f, PER %.5f' % (wer, per))
    print('Total WER(without Stress)%.5f' % (wer_s))


def create_Testdict(file):
    '''Create dictionary mapping word to its different pronounciations.
    '''
    dic = {}
    dic_lines = open(file).read().splitlines()
    for line in dic_lines:
        lst = line.strip().split()
        if len(lst) > 1:
            if lst[0] not in dic:
                dic[lst[0]] = [" ".join(lst[1:])]
            else:
                dic[lst[0]].append(" ".join(lst[1:]))
        elif len(lst) == 1:
            print("WARNING: No phonemes for word '%s' line ignored" % (lst[0]))
    return dic


# def process_target(data_dir):
#     targetTable = np.load(data_dir + "/targetTable.npy").item()
#     rev_ph_vocab = {y: x for x, y in targetTable.items()}
#     return (targetTable, rev_ph_vocab)
#
#
# def process_source(data_dir):
#     sourceTable = np.load(data_dir + "/sourceTable.npy").item()
#     rev_gh_vocab = {y: x for x, y in sourceTable.items()}
#     return (sourceTable, rev_gh_vocab)


@app.route('/decode/<word>')
def decode(word):
    wordlist = word.split()
    output = ""
    for word in wordlist:
        punc_regex = str.maketrans('', '', string.punctuation)
        stripped_word = word.translate(punc_regex)
        output += jsonify(str(decoded_word))
    return(output)


def serve():
    sess = tf.Session()
    # Load model
    with tf.variable_scope("model"):
        model, _ = get_model(sess, isTraining=False)
    return sess, model


# def optimizeModel():
#     global gr_vocab, rev_gr_vocab, ph_vocab, rev_ph_vocab
#     gr_vocab, rev_gr_vocab = process_source(FLAGS["data_dir"])
#     ph_vocab, rev_ph_vocab = process_target(FLAGS["data_dir"])


if __name__ == "__main__":

    FLAGS = parse_cmd()
    FLAGS.update(setModelParameters())
    # optimizeModel()
    if FLAGS['interactive']:
        interactive()
    elif FLAGS['evaluate']:
        evaluate()
    elif FLAGS['serve']:
        sess, model = serve()
        app.run(host=FLAGS["host"], port=FLAGS["port"])
    sys.stdout.close()
