"""Support Vector Machine Module."""
from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
model = clf.fit(X, y)
predicted = model.predict([[0, 0]])
print predicted[0]
