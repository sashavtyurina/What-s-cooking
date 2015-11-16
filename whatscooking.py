__author__ = 'Alex'
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


def cuisine_lookup(required_id, cuisine_dict):
    for c in cuisine_dict.items():
        if c[1] == required_id:
            return c[0]
    return None


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





