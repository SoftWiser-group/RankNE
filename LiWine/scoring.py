#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

__author__ = "Bryan Perozzi"

import numpy
import sys

from collections import defaultdict
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle as skshuffle
import numpy as np

import warnings

warnings.filterwarnings('ignore')


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def gen_label_vec(labels, label_size):
    label_vec = np.zeros((len(labels), label_size), dtype=int)
    for i in range(len(labels)):
        label_vec[i, labels[i]] = 1
    return label_vec


def evaluate(features, labels, dataset):

    print("evaluating {}".format(dataset))

    np.random.seed(1)

    num_shuffles = 10
    all = False

    labels_list = [0]*len(labels)
    labels_size = 0
    for i in labels:
        labels_size = max(max(labels[i]), labels_size)
        labels_list[i] = list(labels[i])



    features = np.array(features)
    labels = gen_label_vec(labels_list, labels_size+1)
    print(features.shape)
    print(labels.shape)
    #==============================================================================================================
    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(num_shuffles):
        shuffles.append(skshuffle(features, labels))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    if all:
        training_percents = numpy.asarray(range(1, 10)) * .1
    else:
        training_percents = [0.05, 0.07, 0.1, 0.5, 0.9]
    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size, :]

            # y_train = [[] for x in range(y_train_.shape[0])]
            #
            # cy = y_train_.tocoo()
            # for i, j in zip(cy.row, cy.col):
            #     y_train[i].append(j)
            #
            # assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:, :]

            # y_test = [[] for _ in range(y_test_.shape[0])]
            #
            # cy = y_test_.tocoo()
            # for i, j in zip(cy.row, cy.col):
            #     y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [np.sum(y_test_[i]) for i in range(y_test_.shape[0])]
            # print('top_k_list', top_k_list[:10])
            preds = clf.predict(X_test, top_k_list)
            # preds = np.array(preds)
            # preds = preds.squeeze()

            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(y_test_, gen_label_vec(preds, labels_size+1), average=average)
                # if dataset == 'citeseer' or dataset == 'wiki':
                #     results['acc'] = accuracy_score(y_test_, gen_label_vec(preds, labels_size+1))

            all_results[train_percent].append(results)

    print('{} {} Results, using embeddings of dimensionality {}'.format(dataset, "multiview", X.shape[1]))
    print('-------------------')
    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        # for index, result in enumerate(all_results[train_percent]):
        #     print('Shuffle #%d:   ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print('Average score:', dict(avg_score))
        print('-------------------')


