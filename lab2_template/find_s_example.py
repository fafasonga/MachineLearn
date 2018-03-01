#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import read_csv

def load_data(path_to_csv, has_header = True):
    '''
    Loads a csv file, the last column is assumed to be the output
    label 'yes'/'no'

    returns: X - numpy array of size (n,m) of input features
             Y - numpy array of output features
    '''

    if has_header:
        data = read_csv(path_to_csv, header='infer')
    else:
        data = read_csv(path_to_csv, header = None)
    data = data.as_matrix()
    X = data[:,1:]
    Y = data[:,0]
    return X,Y

class FindS:

    _rule = None

    def fit(self,training_data, class_label):
        '''
        Chooses the initial hypothesis from the first positive example
        input: training_data - numpy array (n,m), m is the number of features
                                n is the number of training samples
               class_label - numpy array (n,)
        '''

        num_of_attrib = training_data.shape[1]
        num_of_sample = training_data.shape[0]

        hypothesis = self._init_hypothesis(X,Y)

        # the process of fitting the hypothesis of a concept with Find-S
        # algorithm consist of iterating over the training data and
        # verifying whether each of the attributes of the hypothesis satisfy
        # all positive examples.
        #
        # to find the most general hypothesis you need to iterate over each
        # positive example, iterate over all attributes, in case if the attribute
        # of the hypothesis is different from the attribute of the current
        # positive example we generalize the attribute by replacing it with '?'

        # %%% START YOUR CODE HERE
        for i in range(0, X.shape[0]):
            if(Y[i] == 'e'):
                for j in range(0, X.shape[1]):
                    if hypothesis[j] != X[i, j]:
                        hypothesis[j] = '?'


        # %%% END YOUR CODE HERE

        self._rule = hypothesis

    def get_rule(self):
        return self._rule

    def _init_hypothesis(self,training_data,class_label):
        '''
        Chooses the initial hypothesis from the first positive example
        input: training_data - numpy array (n,m), m is the number of features
                                n is the number of training samples
               class_label - numpy array (n,)
        returns: hypothesis - feature vector of the first positive example
        '''
        hypothesis = None

        # here you need to initialize the hypothesis to the values of the
        # first positive training example

        # %%% START YOUR CODE HERE %%%
        # hypothesis = X[0, :]
        for i in range(0, X.shape[0]):
            if(Y[i] == 'e'):
                hypothesis = X[i,:]
                # for j in range(0, X.shape[1]):
                #     hypothesis[j] = X[i, j]
                #     print (hypothesis)


        # %%% END YOUR CODE HERE %%%

        if hypothesis is None:
            raise Exception("No positive example provided")

        return hypothesis

    def predict():
        raise NotImplemented


X, Y = load_data("mushrooms.csv")


find_s = FindS()

find_s.fit(X, Y)

print("Final Hypothesis:" )
print(find_s.get_rule())
