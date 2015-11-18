__author__ = 'Sajin'

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
import math
import operator

# Given all the training data return only the recipes that belong to a given cuisine
def select_cuisine(cuisine_name, all_recipes):
    '''

    :param cuisine_name:
    :param datafilename: filename with json labeled data
    :return: a list of recipes of a given cuisine

    Usage example:
    italian_recipes = select_cuisine('italian', 'train.json')
    '''

    picked_recipes = []
    picked_ingredients = []
    # all_recipes = pd.read_json(datafilename)
    for r in all_recipes.values:
        if r[0] == cuisine_name:
            picked_recipes.append(r)
            picked_ingredients.append(r[2])
    return picked_ingredients

def cross_validate(text_clf, X, Y, folds=10):
    '''
    K-fold cross validation

    :param text_clf: a classifier
    :param Xtrain: trainig set
    :param folds: percentage of training set to be used for testing
    :return:

    Usage example:
    cross_validate(text_clf, train_ingredients, data_train.cuisine, 5)
    '''
    skf = StratifiedKFold(Y, folds, True)
    averaged_error = 0

    for train_indices, test_indices in skf:  # these are indices of train and test sets
        # number of passes should be equal to the number of folds
        error = 0

        Xtrain = [X[i] for i in train_indices]
        Ytrain = [Y[i] for i in train_indices]

        Xtest = [X[i] for i in test_indices]
        Ytest = [Y[i] for i in test_indices]

        trained_model = text_clf.fit(Xtrain, Ytrain)
        predicted = text_clf.predict(Xtest)

        for i in range(0, len(predicted)):
            error += predicted[i] != Ytest[i]
        error /= (len(predicted) + 1)
        print(error)
        averaged_error += error

    averaged_error /= folds
    return averaged_error

def cross_validate_forpython2(text_clf, X, Y, folds=10):
    '''
    K-fold cross validation

    :param text_clf: a classifier
    :param Xtrain: trainig set
    :param folds: percentage of training set to be used for testing
    :return:

    Usage example:
    cross_validate(text_clf, train_ingredients, data_train.cuisine, 5)
    '''
    skf = StratifiedKFold(Y, folds, True)
    averaged_error = 0

    for train_indices, test_indices in skf:  # these are indices of train and test sets
        # number of passes should be equal to the number of folds
        error = 0

        Xtrain = []
        Ytrain = []
        Xtest = []
        Ytest = []

        for i in train_indices:
            Xtrain.append(X[i])

        for i in train_indices:
            Ytrain.append(Y[i])

        for i in test_indices:
            Xtest.append(X[i])

        for i in test_indices:
            Ytest.append(Y[i])

        trained_model = text_clf.fit(Xtrain, Ytrain)
        predicted = text_clf.predict(Xtest)

        for i in range(0, len(predicted)):
            error += predicted[i] != Ytest[i]
        error /= (len(predicted) + 1)
        # print error
        averaged_error += error

    averaged_error /= folds

    # print averaged_error

    return averaged_error

def compute_KLD (Q, P):
    '''
    Returns top n ingredients from corpus P (cuisine) compared to the background model Q (all recipes)
    :param P:
    :param Q:
    :return:
    '''

    Qdict = {}
    for q in Q:
        Qdict[q] = Qdict.get(q, 0) + 1

    Pdict_ = {}
    for p in P:
        Pdict_[p] = Pdict_.get(p, 0) + 1

    italian_ingr = sorted(Pdict_.items(), key=operator.itemgetter(1), reverse=True)
    # print(italian_ingr[:100])

    Pdict = {}
    for p in Pdict_.items():
        if p[1] > 50:
            Pdict[p[0]] = p[1]

    kld_values = {}
    for i in Pdict.items():
        ingredient = i[0]
        if not Qdict.get(ingredient, 0):
            continue
        kld_values[ingredient] = Pdict.get(ingredient, 0) * math.log(Pdict.get(ingredient, 1) / Qdict.get(ingredient, 1))

    sorted_ngr = sorted(kld_values.items(), key=operator.itemgetter(1), reverse=True)
    with open('typical_ingredients.txt', 'a') as f:
        f.write('MEXICAN\n')
        f.write('\n'.join([ii[0] for ii in sorted_ngr[:20]]))
        f.write('\n*****\n')
    # print('\n'.join([ii[0] for ii in sorted_ngr[:20]]))


def unique_ingredients(Q):
    Qdict = {}
    for q in Q:
        Qdict[q] = Qdict.get(q, 0) + 1

    Qdict_trunc = {}
    threshold = 50
    for i in Qdict.items():
        if i[1] < threshold:
            continue
        Qdict_trunc[i[0]] = i[1]

    Qdict_trunc = sorted(Qdict_trunc.items(), key=operator.itemgetter(1), reverse=True)
    print(len(Qdict_trunc))



#Read Json's as DataFrame data_train and data_test
data_train = pd.read_json('train.json')
N_train = np.shape(data_train)[0]
data_test = pd.read_json('test.json')
N_test = np.shape(data_test)[0]
# print('%d, %d ' % (N_train, N_test))

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 5))),
                     #('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, n_iter=10, random_state=7))])



#Converting Training Json List to String for CountVectorizer compatibitlity
train_ingredients = []
all_ingredients = []

for i in data_train.ingredients:
    # train_ingredients.append([l for l in i])

    # concat multiword ingredients
    concatted_ingredients = []
    for ingr in i:
        concatted_ingredients.append('_'.join(ingr.split(' ')))
    train_ingredients.append(str(concatted_ingredients))

    all_ingredients += [l for l in i]

result = text_clf.fit(train_ingredients, data_train.cuisine)

# Typical ingredients
# italian = select_cuisine('mexican', data_train)
# italian_ingredients = sum(italian, [])
# compute_KLD(all_ingredients, italian_ingredients)

# truncate ingredients
unique_ingredients(all_ingredients)

print('done')
input()

cross_validate(text_clf, train_ingredients, data_train.cuisine)


predict_train = text_clf.predict(train_ingredients)



# Converting Test Json Ingredient List to String for CountVectorizer compatibility
test_ingredients = []
for i in data_test.ingredients:
    test_ingredients.append(str(i))
predict_test = text_clf.predict(test_ingredients)

# Print Training Error
err = 0
for i in range(0, N_train):
    if predict_train[i] != data_train.cuisine[i]:
        err += 1
# print('Training Error : %f' % (float(err)/N_train))
# print('%d, %f' % (N_train, err))

# Create submission file
with open('submission.csv', 'w+') as out_file:
    out_file.write('id,cuisine\n')
    for i in range(0, N_test):
        out_file.write(str(data_test.id[i]) + ',' + predict_test[i] + '\n')

# Cross-validation bit
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#     train_ingredients, data_train.cuisine, test_size=0.4, random_state=0)



