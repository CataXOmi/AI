import nltk
import math
import random
import pandas as pd
import numpy as np
from collections import Counter
from clasificator import KNN_classifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import re
import string

def tokenize(text):
    '''
    Generic wrapper around different tokenization methods.
    '''
    return nltk.WordPunctTokenizer().tokenize(text)


def get_corpus_vocabulary(corpus):
    '''
    Write a function to return all the words in a corpus.
    '''

    counter = Counter()
    for text in corpus:

        tokens = tokenize(text)
        counter.update(tokens)

    return counter


def get_representation(toate_cuvintele, how_many):
    '''
    Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wd2idx     @  che  .   ,   di  e
    idx2wd     0   1   2   3   4   5

    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):

        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant

    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):
    '''
    Convert a text to a bag of words representation.
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text   0   1   0   2   0   1
    '''

    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1

    return features

def corpus_to_bow(corpus, wd2idx):
    '''
    Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)

    return all_features

    '''
    #SAU
    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)

    return all_features
    '''



def write_prediction(out_file, predictions):
    '''
    A function to write the predictions to a file.
    id,label
    5001,1
    5002,1
    5003,1
    ...
    '''

    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed


def split(data, labels, procentaj_valid=0.25):
    '''
    Split data and labels into train and valid by procentaj_valid.
    75% train, 25% valid
    Important! shuffle the data before splitting.
    '''
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    N = int((1 - procentaj_valid) * len(labels))
    train = data[indici[:N]]
    valid = data[indici[N:]]
    y_train = labels[indici[:N]]
    y_valid = labels[indici[N:]]

    return train, valid, y_train, y_valid


def cross_validate(k, data, labels):
    '''
    Split the data into k chunks.
    iteration 0:
        chunk 0 is for validation, chunk[1:] for train
    iteration 1:
        chunk 1 is for validation, chunk[0] + chunk[2:] for train
    ...
    iteration k:
        chunk k is for validation, chunk[:k] for train
    '''
    chunk_size = len(labels) // k
    indici = np.arange(0, len(labels))
    random.shuffle(indici)

    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i + chunk_size:]])
        valid = data[valid_indici]
        train = data[train_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid

def precision_recall_score(y_true, y_pred):

    tp = 0
    tf = 0
    fn = 0
    fp = 0
    idx = 0

    while idx < len(y_true) and idx < (len(y_pred)):
        if y_pred[idx] == 1:
            if y_pred[idx] == y_true[idx]:
                tp += 1
        if y_pred[idx] == 0:
            if y_pred[idx] == y_true[idx]:
                tf += 1
        if y_pred[idx] == 1:
            if y_true[idx] == 0:
                fp += 1
        if y_pred[idx] == 0:
            if y_true[idx] == 1:
                fn += 1
        idx += 1

    return tp/(tp+fp), tp/(tp+fn)


def Average(lst):
    return sum(lst) / len(lst)

# Citire date
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Missing Values
print(train_df.isna().sum())

corpus = train_df['text']
print(corpus)

toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)

print(toate_cuvintele)
data = np.array(corpus_to_bow(corpus, wd2idx))
labels = train_df['label'].values

#Normalizare
for idx in range(0, data.shape[1]):
    if np.sum(data[idx]) != 0:
        data[idx] = data[idx]/math.sqrt(np.sum(data[idx])) #(np.sum(data[idx]))

test_data = np.array(corpus_to_bow(test_df['text'], wd2idx))

for idx in range(0, test_data.shape[1]):
    if np.sum(test_data[idx]) != 0:
        test_data[idx] = test_data[idx]/math.sqrt(np.sum(test_data[idx])) #(np.sum(test_data[idx]))

# KNN SKLEARN
clf = KNeighborsClassifier(n_neighbors=17)
acc_list = []
for i in range(10):
    train, valid, y_train, y_valid = split(data, labels)
    clf.fit(train, y_train)
    predictii = clf.predict(valid)
    precision, recall = precision_recall_score(y_valid, predictii)
    f1 = (2*precision*recall)/(precision+recall)
    print('Testarea', i, 'are', f1)
    acc_list.append(f1)

print(acc_list)
average = Average(acc_list)
print('\n', average)


#clasificator knn sklearn
preds= clf.predict(test_data).astype('int')
print(preds)

write_prediction('sample_submission.csv', preds)

