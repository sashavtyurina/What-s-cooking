__author__ = 'Alex'
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import operator
import re


def cuisine_lookup(required_id, cuisine_dict):
    for c in cuisine_dict.items():
        if c[1] == required_id:
            return c[0]
    return None

def ingredient_frequency(inputfilename, outputfilename):
    with open(inputfilename) as f:
        d = eval(f.read())

    frequency = {}
    for item in d:
        ingredients = item['ingredients']
        for i in ingredients:
            frequency[i] = frequency.get(i, 0) + 1

    sorted_freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    with open(outputfilename, 'w') as ff:
        ff.write('name,count\n')
        for item in sorted_freq:
            ff.write("%s,%d\n" % (item[0], item[1]))

def splitAndStem(inputfilename, outputfilename):
    '''
    For each ingredient split it into words, stem each word, construct a new recipe from those words
    :param inputfilename:
    :return:
    '''


    with open(outputfilename, 'w') as ff:
        ff.write('[\n')

    with open(inputfilename) as f:
        d = eval(f.read())

    stemmer = PorterStemmer()
    with open(outputfilename, 'a') as ff:
        for i in d:
            # print(i)
            new_item = {}
            new_ingredients = []
            for ingredient in i['ingredients']:
                tokens = word_tokenize(ingredient)
                clean_tokens = [re.subn('[^A-Za-z]', '', token)[0] for token in tokens]
                new_ingredients += [stemmer.stem(w).lower() for w in clean_tokens]
            new_item['cuisine'] = i['cuisine']
            new_item['id'] = i['id']
            new_item['ingredients'] = new_ingredients
            json_recipe = json.dumps(new_item)
            ff.write('%s,\n' % str(json_recipe))

# splitAndStem('train.json', 'split_ingredients.json')
ingredient_frequency('split_ingredients.json', 'ingredient_frequency.csv')
print('Done counting')
input()


vectorizer = DictVectorizer()

with open('train.json') as f:
    d = eval(f.read())
    cuisines = dict()
    ingredients_train = []
    labels = []
    next_label_id = 0

    for recipe in d:
        recipe_dict = {}

        if recipe['cuisine'] not in cuisines.keys():
            cuisines[recipe['cuisine']] = next_label_id
            next_label_id += 1
        labels.append(cuisines.get(recipe['cuisine'], -1))

        for ingr in recipe['ingredients']:
            recipe_dict[ingr] = recipe_dict.get(ingr, 0) + 1
        ingredients_train.append(recipe_dict)

    X_train = vectorizer.fit_transform(ingredients_train)

d = {}



ingridients_test = []
ids = []
with open('test.json') as ff:
    d = eval(ff.read())

    for recipe in d:
        ingredients_dict = {}
        ids.append(recipe['id'])

        for ingr in recipe['ingredients']:
            ingredients_dict[ingr] = ingredients_dict.get(ingr, 0) + 1
        ingridients_test.append(ingredients_dict)

    X_test = vectorizer.transform(ingridients_test)


# total_corpus = ingridients_test + ingredients_train
# X = vectorizer.fit_transform(total_corpus)
#
# X_test = X[:len(ingridients_test)]
# X_train = X[(len(ingridients_test) + 1):]


# gnb = GaussianNB()
# y_prediction = gnb.fit(X.toarray(), labels).predict(X.toarray())

classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-5, n_iter=100, random_state=5)
# y_prediction = classifier.fit(X.toarray(), labels).predict(X.toarray())
y_prediction = classifier.fit(X_train.toarray(), labels).predict(X_test.toarray())
y_prediction_train = classifier.fit(X_train.toarray(), labels).predict(X_train.toarray())

with open('submission3.txt', 'w') as out:
    out.write('id,cuisine\n')

    for i in range(0, len(ids)):
        out.write('%d,%s\n' % (ids[i], cuisine_lookup(y_prediction[i], cuisines)))



with open('true_labels.txt', 'w') as f:
    for yy in labels:
        f.write('%d\n' % yy)

with open('prediction_train.txt', 'w') as f:
    for yy in y_prediction_train:
        f.write('%d\n' % yy)
# # print('%d' % (y_prediction != labels).sum())
#
errors = 0
y_prediction_train = open('prediction_train.txt').read().split('\n')
labels = open('true_labels.txt').read().split('\n')


for i in range(0, len(labels)):
    errors += y_prediction_train[i] != labels[i]

# print('%d' % (y_prediction != labels).sum())
print(errors/len(labels))





