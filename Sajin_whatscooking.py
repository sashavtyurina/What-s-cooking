__author__ = 'Sajin'

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np

#Read Json's as DataFrame data_train and data_test
data_train = pd.read_json('train.json')
N_train = np.shape(data_train)[0]
data_test = pd.read_json('test.json')
N_test = np.shape(data_test)[0]
print N_train, N_test

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,5))),
                     #('tfidf', TfidfTransformer()),
                     ('clf',SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, n_iter=1000, random_state=7))])


#Converting Training Json List to String for CountVectorizer compatibitlity
train_ingredients = []
for i in data_train.ingredients:
    train_ingredients.append(str(i))
result = text_clf.fit(train_ingredients, data_train.cuisine)
predict_train = text_clf.predict(train_ingredients)

#Converting Test Json Ingredient List to String for CountVectorizer compatibility
test_ingredients = []
for i in data_test.ingredients:
    test_ingredients.append(str(i))
predict_test = text_clf.predict(test_ingredients)

#Print Training Error
err = 0
for i in range(0,N_train):
    if predict_train[i]!=data_train.cuisine[i]:
        err+=1
print 'Training Error : %f' % (float(err)/N_train)
print N_train , err

#Create submission file
with open('submission.csv','w+') as out_file:
    out_file.write('id,cuisine\n')
    for i in range(0,N_test):
        out_file.write(str(data_test.id[i])+','+predict_test[i]+'\n')

