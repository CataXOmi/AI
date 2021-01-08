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
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
import re
import spacy
from spacy.lemmatizer import Lemmatizer
import time

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

    #Version1
    all_features = []

    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)

    return all_features
    '''

    all_features = np.zeros((len(corpus), len(wd2idx)))

    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)

    return all_features


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


def confusion_matrix(y_true, y_pred):
    numar_clase = 2
    conf_matr = np.zeros((numar_clase,numar_clase))
    for idx in range(0,len(y_true)):
        if y_true[idx] == y_pred[idx]:
            conf_matr[y_pred[idx]][y_pred[idx]] += 1
        else:
            conf_matr[y_true[idx]][y_pred[idx]] += 1

    return conf_matr


def medie(lst):
    return sum(lst) / len(lst)


# Citire date
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Missing Values
print(train_df.isna().sum())

# Bag of words transformation
corpus = train_df['text']
toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)

print(len(toate_cuvintele))

data = np.array(corpus_to_bow(corpus, wd2idx))
labels = train_df['label'].values
test_data = np.array(corpus_to_bow(test_df['text'], wd2idx))

print(data.shape)
print(test_data.shape)

# Normalizare
for idx in range(0, data.shape[1]):
    if np.sum(data[idx]) != 0:
        data[idx] = data[idx]/math.sqrt(np.sum(data[idx]))  # (np.sum(data[idx]))

for idx in range(0, test_data.shape[1]):
    if np.sum(test_data[idx]) != 0:
        test_data[idx] = test_data[idx]/math.sqrt(np.sum(test_data[idx]))  # (np.sum(test_data[idx]))

# KNN SKLEARN
clf = KNeighborsClassifier(n_neighbors=17)

f1_list = []
timp_antrenare = []
for i in range(10):
    train, valid, y_train, y_valid = split(data, labels)
    t_start_antrenare = time.time()
    clf.fit(train, y_train)
    t_sfarsit_antrenare = time.time()
    timp_ant = t_sfarsit_antrenare - t_start_antrenare
    predictii = clf.predict(valid)
    precision, recall = precision_recall_score(y_valid, predictii)
    f1 = (2*precision*recall)/(precision+recall)
    conf_matr = confusion_matrix(y_valid, predictii)
    print("Testarea", i+1, "are f1 score", f1, ",timp de antrenare", timp_ant, "si matricea de confuzie:\n", conf_matr)
    f1_list.append(f1)
    timp_antrenare.append(timp_ant)
print("Media f1 scorurilor este", medie(f1_list), ",iar durata media a antrenarii acestui model este", np.mean(timp_antrenare), "si timpul total de antrenare pe clf.fit in cadrul split-ului: ", sum(timp_antrenare))


print("\nCalculul timpului de antrenare pentru documentatie :\n")
f1_list = []
t_start_antrenare = time.time()
for i in range(10):
    train, valid, y_train, y_valid = split(data, labels)
    clf.fit(train, y_train)
    predictii = clf.predict(valid)
    precision, recall = precision_recall_score(y_valid, predictii)
    f1 = (2*precision*recall)/(precision+recall)
    conf_matr = confusion_matrix(y_valid, predictii)
    print("Testarea", i, "are f1 score", f1, "si matricea de confuzie:\n", conf_matr)
    f1_list.append(f1)
t_sfarsit_antrenare = time.time()
timp_ant = t_sfarsit_antrenare - t_start_antrenare
print("Media f1 scorurilor este", medie(f1_list), ",iar durata antrenarii acestui model este", timp_ant)


print('\n')
preds = clf.predict(test_data).astype('int')
print(preds)

write_prediction('sample_submission.csv', preds)

