import nltk
import pandas as pd
import numpy as np
from collections import Counter


TRAIN_FILE = ''
TEST_FILE = ''
TXT_COL = 'text'
LBL_COL = 'label'


def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    return nltk.WordPunctTokenizer().tokenize(text)
    #return nltk.TweetTokenizer().tokenize(text)

def get_representation(vocabulary, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wrd: @  che  .   ,   di  e
    idx: 0   1   2   3   4   5
    '''
    most_comm = vocabulary.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    # complete code
    # un for prin vocabular w2idx[cuvant] = poztie
    # id2wd[pozitie] = cuvant

    for i, iterator in enumerate(most_comm):
        cuv = iterator[0]
        wd2idx[cuv] = i
        idx2wd[i] = cuv

    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    tokenz = tokenize(text)
    for tok in tokenz:
        if tok in wd2idx:
            features[wd2idx[tok]] += 1


    return features


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2

    '''

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text,wd2idx))

    all_features = np.array(all_features)
    return all_features


def write_prediction(out_file, predictions):
    '''A function to write the predictions to a file.
    '''
    pass


def main():
    pass

if __name__ == '__main__':
    main()

#### citim fisier csv

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

text = train_df['text'][2]



'''
text = 'https://gosad/com @Lala #super code'
print(text.split(' '))
print(tokenize(text))
#print(train_df['text'][16]) # text
#print(train_df['label'][16]) # eticheta

#print(test_df)

from nltk.tokenize import TweetTokenizer
tknz = TweetTokenizer(reduce_len=True) # string_handles
tokens = tknz.tokenize(text)
print(tokens)

#numara de cate ori apare un cuvant in text
ctr = {}
for word in tokens:
    if word in ctr:
        ctr[word] += 1
    else:
        ctr[word] = 1
print(ctr)

#sau

from collections import Counter

ctr = Counter()
ctr.update(tokens)
print(ctr.most_common(3))
'''

toate_cuvintele = get_corpus_vocabulary(corpus)
print(toate_cuvintele)
print(toate_cuvintele.most_common(10))
wd2idx, idx2wd = get_representation(toate_cuvintele, 10)

print(text_to_bow(text,wd2idx))
print('\n',corpus_to_bow(corpus,wd2idx))

data = corpus_to_bow(corpus,wd2idx)
labels = train_df['label']

test_data = corpus_to_bow(test_df['text'],wd2idx)

from clasificator import KNN_classifier
clf = KNN_classifier(data, labels)
predictii = clf.classify_images()

#random split pe data si labels le impartim pentru a putea face teste desi nu avem ala pe kaggle