# -*- coding: utf-8 -*-
# Run the program "python data_processing_syllabification.py -d CELEX_IPA_data_syllabification.txt" 

import argparse
import collections
import pathlib
import os.path
import re
import random
import csv

import numpy as np
from sklearn.model_selection import train_test_split

# Special vocabulary symbols - we always put them at the start.

_PAD = "_PAD"  # padding required for a sequence to match mini-batch length
_SOS = "_SOS"  # START symbol for decoding [ SOS = Start of Sentence]
_EOS = "_EOS"  # END synbol for decoding [EOS = End of Sentence]

_START_VOCAB = [_PAD, _SOS, _EOS]
_buckets = [(35, 35)]

# for random buckets
# _buckets = [(5, 10), (10, 15), (40, 50)]

global data_dir
data_dir=""


def processLine(line):
    """
    Process a single line give as input in the following format

    hello HH AH0 L OW1

    [word space phoneme_set (separated by space)]

    Words with these characters ['().0123456789'] e.g. HOUSE(2) are removed

    :param
        line: a line form the cmu_data file

    :return:
        graphemeSet: list chars in the word
        phonemeSet:  list phonemes in the pronunciation
    """

    line = line.strip()
    grapheme, phoneme = line.split(",", 1)
    grapheme = grapheme.strip()
    phoneme = phoneme.strip()
    graphemeSet = list(grapheme)
    phonemeSet = phoneme.split(" ")

    return graphemeSet, phonemeSet


def readDataSet(dictFile):
    """
    Process dataset provided as commandline argument
    int two sets of ordered dict of gramphemeDict and
    phonemeTable

    :param:

        dictFile: a dictionary file containing grapheme to phoneme examples

    :return:
        graphemeTable: a indexed ordered dictionary of all the graphemes
        phonemeTable: a indexed ordered dictionary of all the phonemes
    """
    graphemeCounter = 0
    phonemeCounter = 0

    graphemeTable = collections.OrderedDict()
    phonemeTable = collections.OrderedDict()

    graphemeList = []
    phonemeList = []

    data_dir = os.path.dirname(os.path.abspath(dictFile))



    if os.path.exists(dictFile):
        file = open(dictFile, "r", encoding='utf-8')
        for line in file.readlines():
            graphemeSet, phonemeSet = processLine(line)
            graphemeList.append(graphemeSet)
            phonemeList.append(phonemeSet)

        # Remove words with these characters e.g. HOUSE(2)
        redundant = '().0123456789'
        redundant_word = re.compile('[%s]' % redundant)

        # Removing redundancies and abnormal words from the cmu_data
        graphemeList, phonemeList = zip(*[(x, y) for x, y in zip(graphemeList, phonemeList)
                                          if not bool(redundant_word.findall(''.join(x)))])

        for token in _START_VOCAB:
            graphemeTable[token] = graphemeCounter
            graphemeCounter += 1
            phonemeTable[token] = phonemeCounter
            phonemeCounter += 1

        for grapheme in graphemeList:
            for literal in grapheme:
                if literal not in graphemeTable:
                    graphemeTable[literal] = graphemeCounter
                    graphemeCounter += 1

        for phoneme in phonemeList:
            for phone in phoneme:
                if phone not in phonemeTable:
                    phonemeTable[phone] = phonemeCounter
                    phonemeCounter += 1

    else:
        logging.debug("Can't find file: %s", dictFile)


    graphemeTableFile = data_dir + "/graphemeTable.csv"
    phonemeTableFile = data_dir + "/phonemeTable.csv"
    
    g = csv.writer(open(graphemeTableFile, "w", encoding='utf-8'))
    for key, val in graphemeTable.items():
        g.writerow([key, val])


    p = csv.writer(open(phonemeTableFile, "w", encoding='utf-8'))
    for key, val in phonemeTable.items():
        p.writerow([key, val])

    # assert (graphemeTable == np.genfromtxt(graphemeTableFile, delimiter=','))
    # assert (phonemeTable == np.genfromtxt(phonemeTableFile, delimiter=','))

    graphemeList_train, graphemeList_test, phonemeList_train, phonemeList_test = train_test_split(graphemeList,
                                                                                                  phonemeList,
                                                                                                  test_size=0.15,
                                                                                                  random_state=42)

    graphemeList_test, graphemeList_val, phonemeList_test, phonemeList_val = train_test_split(graphemeList_test,
                                                                                              phonemeList_test,
                                                                                              test_size=0.2,
                                                                                              random_state=35)

    return graphemeList_train, graphemeList_val, graphemeList_test, phonemeList_train, phonemeList_val, phonemeList_test, graphemeTable, phonemeTable, data_dir


def processAndSaveData(graphemeList, phonemeList, graphemeTable, phonemeTable, dataType,data_dir):
    """

    :param graphemeList:    List of Graphemes in the dataSet
    :param phonemeList:     List of Phonemes in the dataSet
    :param graphemeTable:   Table of each possible chars encountered in the dataSet
    :param phonemeTable:    Table of each possible phones encountered in the dataSet
    :param dataType:        Type in which data needs to be saved for exmaple (Train, Test, Validation)

    """
    graphemeIds = []
    phonemeIds = []

    graphemeFile = data_dir + "/grapheme_" + dataType
    phonemeFile = data_dir + "/phoneme_" + dataType

    saveSplittedGraphemeLoc = data_dir + "/readableData/"
    saveSplittedPhonemeLoc = data_dir + "/readableData/"

    pathlib.Path(saveSplittedGraphemeLoc).mkdir(parents=True, exist_ok=True)
    with open(saveSplittedGraphemeLoc + dataType + "_grapheme.txt", "w", encoding='utf-8') as f:
        for grapheme in graphemeList:
            graphemeIds.append([graphemeTable[token] for token in grapheme])
            f.write(str(grapheme)+"\n")

    f.close()

    pathlib.Path(saveSplittedPhonemeLoc).mkdir(parents=True, exist_ok=True)
    with open(saveSplittedPhonemeLoc + dataType + "_phoneme.txt", "w", encoding='utf-8') as f:
        for phoneme in phonemeList:
            phonemeIds.append([phonemeTable[phone] for phone in phoneme])
            f.write(str(phoneme)+"\n")
    f.close()

    np.save(graphemeFile, graphemeIds)
    np.save(phonemeFile, phonemeIds)



def argumentParsing():
    """
    Parsing command line arguments
    :return: argument table
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dict", type=str, help="Dictonary File location")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    FLAGS = argumentParsing()
    graphemeList_train, graphemeList_val, graphemeList_test, phonemeList_train, phonemeList_val, phonemeList_test, graphemeTable, phonemeTable, data_dir = readDataSet(FLAGS.dict)

    processAndSaveData(graphemeList_train, phonemeList_train, graphemeTable, phonemeTable, "train", data_dir)
    print("Generated Train set")
    processAndSaveData(graphemeList_val, phonemeList_val, graphemeTable, phonemeTable, "validation",data_dir)
    print("Generated Validation set")
    processAndSaveData(graphemeList_test, phonemeList_test, graphemeTable, phonemeTable, "test",data_dir)
    print("Generated Test set")