import nltk
import math
import random
import pandas as pd
import time
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


lemmatizer = WordNetLemmatizer()
nlp = spacy.load('it_core_news_lg')
stemmer = SnowballStemmer(language='italian')
stop_words = set(stopwords.words('italian'))


def tokenize(text):
    '''
    Generic wrapper around different tokenization methods.
    '''

    # text = nlp(text)
    text = str(text)
    text = text.lower()
    text = text.strip() # stergem white space uri
    #text = text.replace('{html}', "")
    text = re.sub(r'@[A-Z0-9a-z_:.></?;|!@#%^($&)=+,]+', '', text)
    text = re.sub(r'#[A-Z0-9a-z_:.></?;|!@#%^($&)=+,]+', '', text)
    text = re.sub(r'http\S+', '', text)
    #text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'\d+', '', text)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    filtered_words = [w for w in tokens if len(w) > 3]  # if not w in stopwords.words('italian')]
    stem_words = [stemmer.stem(w) for w in filtered_words]

    # lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    # lemma_words = [t.lemma_ for t in filtered_words]

    return filtered_words


def get_representation(toate_cuvintele, how_many):
    '''
    Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
           @  che  .   ,   di  e
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


def get_corpus_vocabulary(corpus):
    '''
    Write a function to return all the words in a corpus.
    '''

    counter = Counter()

    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)

    return counter


def text_to_bow(text, wd2idx):
    '''
    Convert a text to a bag of words representation.
           @  che  .   ,   di  e
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
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    all_features = []

    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)

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
    75% train, 25% valid
    important! mai intai facem shuffle la date
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
        valid_indici = indici[i:i+chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i+chunk_size:]])
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
    for idx in range(0, len(y_true)):
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
wd2idx, idx2wd = get_representation(toate_cuvintele, 1000)

print(len(toate_cuvintele))

data = np.array(corpus_to_bow(corpus, wd2idx))  # desi nu era necesar np.array, l-am folosit pentru siguranta
labels = train_df['label'].values
test_data = np.array(corpus_to_bow(test_df['text'], wd2idx))  # desi nu era necesar np.array, l-am folosit pentru siguranta

print(data.shape)
print(test_data.shape)

'''
# Normalizare
for idx in range(0, data.shape[1]):
    if np.sum(data[idx]) != 0:
        data[idx] = data[idx]/math.sqrt(np.sum(data[idx])) #(np.sum(data[idx]))

for idx in range(0, test_data.shape[0]):
    if np.sum(test_data[idx]) != 0:
        test_data[idx] = test_data[idx]/math.sqrt(np.sum(test_data[idx]))  #(np.sum(test_data[idx]))
'''

clf = MultinomialNB()

'''
# Am comentat aceasta bucata deoarece voiam sa ramana 10 fold-ul necesar documentatiei
nr_epoci = 10
for nr in range(nr_epoci):
    scoruri_ep = []
    i = 1
    timp_antrenare_ep = []
    final_conf_matr = np.zeros((2, 2))
    for train, valid, y_train, y_valid in cross_validate(10, data, labels):
        t_start_antrenare_ep = time.time()
        clf.fit(train, y_train)
        t_sfarsit_antrenare_ep = time.time()
        timp_ant_ep = t_sfarsit_antrenare_ep - t_start_antrenare_ep
        predictii = clf.predict(valid)
        score_ep = f1_score(y_valid, predictii)
        conf_matr = confusion_matrix(y_valid, predictii)
        final_conf_matr += conf_matr
        print("Testarea", i, "are f1 score:", score_ep,  ",timp de antrenare:", timp_ant_ep, "si matricea de confuzie:\n", conf_matr)
        i = i+1
        scoruri_ep.append(score_ep)
        timp_antrenare_ep.append(timp_ant_ep)
    print("\nMedia f1 scorurilor este:", np.mean(scoruri_ep), 'cu deviatia standard:', np.std(scoruri_ep), ",media timpilor de antrenare:", np.mean(timp_antrenare_ep), "si timpul total de antrenare pe clf.fit in cadrul 10 fold: ", sum(timp_antrenare_ep))
    print('\n')
    print("Matricea de confuzie per epoca este:\n", final_conf_matr)
'''

print("Calculul timpului de antrenare pentru documentatie apeland o singura data 10 fold:\n")
scoruri_fold = []
i = 1
timp_antrenare_fold = []
final_conf_matr = np.zeros((2, 2))
for train, valid, y_train, y_valid in cross_validate(10, data, labels):
    t_start_antrenare_fold = time.time()
    clf.fit(train, y_train)
    t_sfarsit_antrenare_fold = time.time()
    timp_ant_fold = t_sfarsit_antrenare_fold - t_start_antrenare_fold
    predictii = clf.predict(valid)
    score_fold = f1_score(y_valid, predictii)
    conf_matr = confusion_matrix(y_valid, predictii)
    final_conf_matr += conf_matr
    print("Testarea", i, "are f1 score:", score_fold, ",timp de antrenare:", timp_ant_fold, "si matricea de confuzie:\n", conf_matr)
    i = i+1
    scoruri_fold.append(score_fold)
    timp_antrenare_fold.append(timp_ant_fold)
print("\nMedia f1 scorurilor este:", np.mean(scoruri_fold), 'cu deviatia standard:', np.std(scoruri_fold), ",media timpilor de antrenare:", np.mean(timp_antrenare_fold), "si timpul total de antrenare pe clf.fit in cadrul 10 fold: ", sum(timp_antrenare_fold))
print('\n')
print("Matricea de confuzie este:\n", final_conf_matr)

print("Calculul timpului de antrenare pentru documentatie apeland o singura data 10 fold:\n")
scoruri_fold = []
i = 1
final_conf_matr = np.zeros((2, 2))
t_start_antrenare_fold = time.time()
for train, valid, y_train, y_valid in cross_validate(10, data, labels):
    clf.fit(train, y_train)
    predictii = clf.predict(valid)
    score_fold = f1_score(y_valid, predictii)
    conf_matr = confusion_matrix(y_valid,predictii)
    final_conf_matr += conf_matr
    print("Testarea", i, "are f1 score:", score_fold, "si matricea de confuzie:\n", conf_matr)
    i = i+1
    scoruri_fold.append(score_fold)
t_sfarsit_antrenare_fold = time.time()
timp_ant_fold = t_sfarsit_antrenare_fold - t_start_antrenare_fold
print("\nMedia f1 scorurilor este:", np.mean(scoruri_fold), 'cu deviatia standard:', np.std(scoruri_fold), "si  timpul de antrenare:", timp_ant_fold)
print('\n')
print("Matricea de confuzie este:\n", final_conf_matr)


t_start_predictie = time.time()
preds = clf.predict(test_data).astype('int')
t_sfarsit_predictie = time.time()


print("Durata de realizare a predictiei : ", t_sfarsit_predictie-t_start_predictie,"\n")
print(preds)

write_prediction('sample_submission6.csv', preds)
