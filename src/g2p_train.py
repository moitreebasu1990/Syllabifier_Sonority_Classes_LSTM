# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Hides verbose log messages
import sys
import csv
import time
import argparse
import random
import json
import numpy as np
import editdistance as ed
import seq2seq_model

import tensorflow as tf

FLAGS = object()
_buckets = [(35, 35)]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

best_wer = 1.0

gr_vocab = None
ph_vocab = None
rev_gr_vocab = None
rev_ph_vocab = None
dict_target_property={}
list_target_property_weights=[]

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

    parser.set_defaults(train=False)
    parser.set_defaults(test=False)

    parser.add_argument("-cd", "--checkpoint_dir",
                        default="./model_checkpoints",type=str,
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


def batch_bucketed_data(bucketed_data, batch_size=64, shuffle=True):
    """Create batches in already bucketed data."""
    DEFAULT_BATCH_SIZE = 64
    if batch_size <= 0:
        print ("Assuming a batch size of %d since specified value is <= 0" % (DEFAULT_BATCH_SIZE))
        batch_size = DEFAULT_BATCH_SIZE

    batched_data = []
    for cur_bucket_data in bucketed_data:
        if shuffle:
            # Shuffle data in each bucket
            random.shuffle(cur_bucket_data)
        batched_data += [cur_bucket_data[i:i + batch_size] for i in xrange(0, len(cur_bucket_data), batch_size)]
    if shuffle:
        # Shuffle the created batches
        random.shuffle(batched_data)
    return batched_data


def loadAndBucketData(dir,type):
    bucket_data = [[] for _ in _buckets]
    test_source = np.load(dir+"/source_"+type+".npy")
    test_target = np.load(dir+"/target_"+type+".npy")

    for i in range(len(test_source)):
        for bucket_id, temp in enumerate(_buckets):
            bucket_data[bucket_id].append((test_source[i], test_target[i]))
            break

    return bucket_data


def setModelParameters():
    """
    Reading model parameters from a json file

    :return parameters: a dictionary containing all the model parameters
    """
    with open('parameters.json', 'r') as fp:
        parameters = json.load(fp)

    source_vocab, _ = initialize_vocabulary(parameters['source_vocab_path'])
    target_vocab, _ = initialize_vocabulary(parameters['target_vocab_path'])

    parameters['source_vocab_size'] = len(source_vocab)
    parameters['target_vocab_size'] = len(target_vocab)


    return parameters


def model_graph(session, training):
    """Create the model graph by creating an instance of Seq2SeqModel."""
    return seq2seq_model.Seq2SeqModel(
        _buckets, training, FLAGS['max_gradient_norm'], FLAGS['batch_size'],
        FLAGS['learning_rate'], FLAGS['learning_rate_decay_factor'],
        FLAGS)


def get_model(session, training):
    model = model_graph(session, training=training)
    mckpts = tf.train.get_checkpoint_state(FLAGS['model_checkpoint_path'])

    steps_done = 0
    try:
        # Restore model parameters
        model.saver.restore(session, mckpts.model_checkpoint_path)
        sys.stdout.write("Loading model parameters from %s\n" %mckpts.model_checkpoint_path)
        sys.stdout.flush()
        steps_done = int(mckpts.model_checkpoint_path.split('-')[-1])
        print("loaded from %d completed steps" % (steps_done))
    except:
        sys.stdout.write("Creating a fresh model with defined parameters.\n")
        sys.stdout.flush()
        # Initialize model parameters
        session.run(tf.global_variables_initializer())
    return model, steps_done


def calc_loss(model, sess, eval_set):
    """Calculate the actual loss function for G2P.

    from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

    Args:
        model: Seq2SeqModel instance
        sess: Tensorflow session with the model compuation graph
        eval_set: Bucketed evaluation set
    Returns:
        wer: Word Error Rate
        per: target Error Rate
    """
    total_words = 0
    total_targets = 0
    wer = 0
    per = 0
    edit_distances = []

    for bucket_id in xrange(len(_buckets)):
        cur_data = eval_set[bucket_id]
        for batch_offset in xrange(0, len(cur_data), FLAGS['batch_size']):
            batch = cur_data[batch_offset:batch_offset + FLAGS['batch_size']]
            num_instances = len(batch)
            # Each instance is a pair of ([Input sequence], [Output sequence])
            inp_ids = [inst[0] for inst in batch]
            gt_ids = [inst[1] for inst in batch]
            encoder_inputs, seq_len, decoder_inputs, seq_len_target = \
                model.get_batch(batch, bucket_id=bucket_id)
            # Run the model to get output_logits of shape TxBx|V|
            output_logits = model.step(sess, encoder_inputs, seq_len,
                                       decoder_inputs, seq_len_target)

            # This is a greedy decoder and output is just argmax at each timestep
            outputs = np.argmax(output_logits, axis=1)
            # Reshape the output and make it batch major via transpose
            outputs = np.reshape(outputs, (max(seq_len_target), num_instances)).T
            for idx in xrange(num_instances):
                cur_output = list(outputs[idx])
                if EOS_ID in cur_output:
                    cur_output = cur_output[:cur_output.index(EOS_ID)]

                gt = gt_ids[idx]
                # Calculate the edit distance from ground truth
                distance = ed.eval(gt, cur_output)
                edit_distances.append((inp_ids[idx], distance, len(gt)))

    edit_distances.sort()

    # Aggregate the edit distances for each word
    word_to_edit = {}
    for edit_distance in edit_distances:
        word, distance, num_targets = edit_distance
        word = tuple(word)
        if word in word_to_edit:
            word_to_edit[word].append((distance, num_targets))
        else:
            word_to_edit[word] = [(distance, num_targets)]

    total_words = len(word_to_edit)
    for word in word_to_edit:
        # Pick the ground truth that's closest to output since their can be
        # multiple pronunciations
        distance, num_targets = min(word_to_edit[word])
        if distance != 0:
            wer += 1
            per += distance
        total_targets += num_targets
    try:
        wer = float(wer)/float(total_words)
    except ZeroDivisionError:
        print ("0 words in evaluation set")
        wer = 1.0
    try:
        per = float(per)/float(total_targets)
    except ZeroDivisionError:
        print ("0 phones in evaluation set")
        per = 1.0
    return wer, per


def train():
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,
                                   inter_op_parallelism_threads=2,log_device_placement=True)) as sess:
        with tf.variable_scope("model", reuse=None):
            # Creates the training graph
            model, steps_done = get_model(sess, training=True)
        with tf.variable_scope("model", reuse=True):
            # Creates the validation graph that reuses training parameters
            mvalid = model_graph(sess, training=False)

        print("Models created")
        print("Reading data from %s" % FLAGS['data_dir'])
        sys.stdout.flush()

        # Load train and dev data
        train_data = loadAndBucketData(FLAGS['data_dir'], "train")
        dev_set = loadAndBucketData(FLAGS['data_dir'], "validation")

        step_time, loss = 0.0, 0.0
        previous_losses = []

        epoch_id = model.epoch.eval()

        val_wer_window = []
        window_size = 3

        # if steps_done > 0:
        #     # The model saved would have wer and per better than 1.0
        #     best_wer = best_wer
        # else:
        #     best_wer = 1.0

        print("Starting training !!\n")
        sys.stdout.flush()
        while (epoch_id < FLAGS['max_epochs']):
            steps = 0.0
            # Batch the data (Also shuffles the data)
            batch_data = batch_bucketed_data(train_data, batch_size=FLAGS['batch_size'])
            for batch in batch_data:
                # Run a minibatch update and record the run times
                start_time = time.time()
                encoder_inputs, seq_len, decoder_inputs, seq_len_target = model.get_batch(batch)
                _, step_loss = model.step(sess, encoder_inputs, seq_len,decoder_inputs, seq_len_target)

                step_time += (time.time() - start_time)
                loss += step_loss

                steps += 1.0
                # print(steps," out of ", len(batch_data), "steps in ", step_time)

            # Increase the epoch counter
            epoch_id += 1
            sess.run(model.epoch_incr)

            step_time /= steps
            loss /= steps
            perplexity = np.exp(loss) if loss < 300 else float('inf')
            print("Epoch %d global step %d learning rate %.4f step-time %.2f"
                  " perplexity %.4f" % (epoch_id, model.global_step.eval(),
                                        model.learning_rate.eval(), step_time,
                                        perplexity))
            if len(previous_losses) >= 3 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            step_time, loss = 0.0, 0.0

            # Calculate validation result
            print("calculating loss now")
            val_wer, val_per = calc_loss(mvalid, sess, dev_set)
            print("Validation WER: %.5f, PER: %.5f" % (val_wer, val_per))
            sys.stdout.flush()

            # Validation WER is a moving window, we add the new entry and pop
            # the oldest one
            val_wer_window.append(val_wer)
            if len(val_wer_window) > window_size:
                val_wer_window.pop(0)
            avg_wer = sum(val_wer_window) / float(len(val_wer_window))
            print("Average Validation WER %.5f" % (avg_wer))
            sys.stdout.flush()

            # The best model is decided based on average validation WER to
            # remove noisy cases of one off validation success
            global best_wer
            if best_wer > avg_wer:
                # Save the best model
                best_wer = avg_wer
                print("Saving Updated Model")
                sys.stdout.flush()
                checkpoint_path = os.path.join(FLAGS['model_checkpoint_path'], "g2p.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step,
                                 write_meta_graph=False)
            print


def test():
    with tf.Session() as sess:
        # Load model
        with tf.variable_scope("model"):
            model, _ = get_model(sess, training=False)
            test_set = loadAndBucketData(FLAGS['data_dir'],"test")
            wer, per = calc_loss(model, sess, test_set)
            print('Total WER %.5f, PER %.5f' % (wer, per))


if __name__ == "__main__":

    FLAGS = parse_cmd()
 
    FLAGS.update(setModelParameters())
    sys.stdout = open("run.log", "w")
    if FLAGS['train']:
        train()
    elif FLAGS['test']:
        test()
    sys.stdout.close()