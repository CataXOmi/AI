import numpy as np


def dist_euclid(img1, img2):
    '''
    :param img1: imagine liniarizata
    :param img2: imagine liniarizata
    :return: distanta euclidiana
    '''
    # atunci cand lucram cu imagini, este important sa facem
    # operatiile in float64, nu in format uint8
    diferente = img1.astype(np.float64) - img2.astype(np.float64)
    diferente = diferente ** 2
    distanta = np.sqrt(np.sum(diferente))
    return distanta


def dist_l1(img1, img2):
    '''
    :param img1: imagine liniarizata
    :param img2: imagine liniarizata
    :return: distanta l1
    '''
    # atunci cand lucram cu imagini, este important sa facem
    # operatiile in float64, nu in format uint8
    diferente = img1.astype(np.float64) - img2.astype(np.float64)
    diferente = np.abs(diferente)
    distanta = np.sum(diferente)
    return distanta


def dist_euclid_array(img, array_de_imagini):
    '''
    :param img1: imagine liniarizata
    :param array_de_imagini: array de imagini liniarizate
    :return: distantele euclidiene intre img si toate imaginile din array
    '''
    # atunci cand lucram cu imagini, este important sa facem
    # operatiile in float64, nu in format uint8
    diferente = img.astype(np.float64) - array_de_imagini.astype(np.float64)
    diferente = diferente ** 2
    distante = np.sqrt(np.sum(diferente, axis=1))
    return distante


def dist_l1_array(img, array_de_imagini):
    '''
    :param img: imagine de test
    :param array_de_imagini: array de imagini
    :return: distantele intre img si array-ul de imagini
    '''
    # atunci cand lucram cu imagini, este important sa facem
    # operatiile in float64, nu in format uint8
    diferente = img.astype(np.float64) - array_de_imagini.astype(np.float64)
    diferente = np.abs(diferente)
    distante = np.sum(diferente, axis=1)
    return distante


class KNN_classifier:
    '''Clasificator KNN simplu.
    '''
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, img_test, k=71, metric='l1'):
        '''Asta e o functie care returneaza predictia pentru img_test
        '''
        if metric == 'l2':
            dst = dist_euclid_array(img_test, self.train_images)
        else:
            dst = dist_l1_array(img_test, self.train_images)
        primele_k_pozitii = dst.argsort()[:k]
        primele_k_etichete = self.train_labels[primele_k_pozitii]
        vect_de_frecv = np.bincount(primele_k_etichete)
        eticheta_prezisa = vect_de_frecv.argmax()
        return eticheta_prezisa

    def classify_images(self, imagini_test, k=71, metric='l1'):
        '''Returneaza toate predictiile pt toate imaginile de test.
        '''
        preds = np.zeros(imagini_test.shape[0])
        for idx in range(0, imagini_test.shape[0]):
            preds[idx] = self.classify_image(imagini_test[idx, :], k=k, metric=metric)
        return preds