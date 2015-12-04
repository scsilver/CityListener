import numpy as np
import pdb

X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
pdb.set_trace()

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
pdb.set_trace()
print(clf.predict(X[2:3]))
pdb.set_trace()
