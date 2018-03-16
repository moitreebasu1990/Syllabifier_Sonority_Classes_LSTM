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

    [word space target_set (separated by space)]

    Words with these characters ['().0123456789'] e.g. HOUSE(2) are removed

    :param
        line: a line form the cmu_data file

    :return:
        sourceSet: list chars in the word
        targetSet:  list targets in the pronunciation
    """

    line = line.strip()
    source, target = line.split(",", 1)
    source = source.strip()
    target = target.strip()
    sourceSet = list(source)
    targetSet = target.split(" ")

    return sourceSet, targetSet


def readDataSet(dictFile):
    """
    Process dataset provided as commandline argument
    int two sets of ordered dict of gramphemeDict and
    targetTable

    :param:

        dictFile: a dictionary file containing source to target examples

    :return:
        sourceTable: a indexed ordered dictionary of all the sources
        targetTable: a indexed ordered dictionary of all the targets
    """
    sourceCounter = 0
    targetCounter = 0

    sourceTable = collections.OrderedDict()
    targetTable = collections.OrderedDict()

    sourceList = []
    targetList = []

    data_dir = os.path.dirname(os.path.abspath(dictFile))



    if os.path.exists(dictFile):
        file = open(dictFile, "r", encoding='utf-8')
        for line in file.readlines():
            sourceSet, targetSet = processLine(line)
            sourceList.append(sourceSet)
            targetList.append(targetSet)

        # Remove words with these characters e.g. HOUSE(2)
        redundant = '().0123456789'
        redundant_word = re.compile('[%s]' % redundant)

        # Removing redundancies and abnormal words from the cmu_data
        sourceList, targetList = zip(*[(x, y) for x, y in zip(sourceList, targetList)
                                          if not bool(redundant_word.findall(''.join(x)))])

        for token in _START_VOCAB:
            sourceTable[token] = sourceCounter
            sourceCounter += 1
            targetTable[token] = targetCounter
            targetCounter += 1

        for source in sourceList:
            for literal in source:
                if literal not in sourceTable:
                    sourceTable[literal] = sourceCounter
                    sourceCounter += 1

        for target in targetList:
            for phone in target:
                if phone not in targetTable:
                    targetTable[phone] = targetCounter
                    targetCounter += 1

    else:
        logging.debug("Can't find file: %s", dictFile)


    sourceTableFile = data_dir + "/sourceTable.csv"
    targetTableFile = data_dir + "/targetTable.csv"
    
    g = csv.writer(open(sourceTableFile, "w", encoding='utf-8'))
    for key, val in sourceTable.items():
        g.writerow([key, val])


    p = csv.writer(open(targetTableFile, "w", encoding='utf-8'))
    for key, val in targetTable.items():
        p.writerow([key, val])

    # assert (sourceTable == np.genfromtxt(sourceTableFile, delimiter=','))
    # assert (targetTable == np.genfromtxt(targetTableFile, delimiter=','))

    sourceList_train, sourceList_test, targetList_train, targetList_test = train_test_split(sourceList,
                                                                                                  targetList,
                                                                                                  test_size=0.15,
                                                                                                  random_state=42)

    sourceList_test, sourceList_val, targetList_test, targetList_val = train_test_split(sourceList_test,
                                                                                              targetList_test,
                                                                                              test_size=0.2,
                                                                                              random_state=35)

    return sourceList_train, sourceList_val, sourceList_test, targetList_train, targetList_val, targetList_test, sourceTable, targetTable, data_dir


def processAndSaveData(sourceList, targetList, sourceTable, targetTable, dataType,data_dir):
    """

    :param sourceList:    List of sources in the dataSet
    :param targetList:     List of targets in the dataSet
    :param sourceTable:   Table of each possible chars encountered in the dataSet
    :param targetTable:    Table of each possible phones encountered in the dataSet
    :param dataType:        Type in which data needs to be saved for exmaple (Train, Test, Validation)

    """
    sourceIds = []
    targetIds = []

    sourceFile = data_dir + "/source_" + dataType
    targetFile = data_dir + "/target_" + dataType

    saveSplittedsourceLoc = data_dir + "/readableData/"
    saveSplittedtargetLoc = data_dir + "/readableData/"

    pathlib.Path(saveSplittedsourceLoc).mkdir(parents=True, exist_ok=True)
    with open(saveSplittedsourceLoc + dataType + "_source.txt", "w", encoding='utf-8') as f:
        for source in sourceList:
            sourceIds.append([sourceTable[token] for token in source])
            f.write(str(source)+"\n")

    f.close()

    pathlib.Path(saveSplittedtargetLoc).mkdir(parents=True, exist_ok=True)
    with open(saveSplittedtargetLoc + dataType + "_target.txt", "w", encoding='utf-8') as f:
        for target in targetList:
            targetIds.append([targetTable[phone] for phone in target])
            f.write(str(target)+"\n")
    f.close()

    np.save(sourceFile, sourceIds)
    np.save(targetFile, targetIds)



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
    sourceList_train, sourceList_val, sourceList_test, targetList_train, targetList_val, targetList_test, sourceTable, targetTable, data_dir = readDataSet(FLAGS.dict)

    processAndSaveData(sourceList_train, targetList_train, sourceTable, targetTable, "train", data_dir)
    print("Generated Train set")
    processAndSaveData(sourceList_val, targetList_val, sourceTable, targetTable, "validation",data_dir)
    print("Generated Validation set")
    processAndSaveData(sourceList_test, targetList_test, sourceTable, targetTable, "test",data_dir)
    print("Generated Test set")