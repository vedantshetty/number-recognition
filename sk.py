import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm#will categorize the data for us

digits = datasets.load_digits()
clf = svm.SVC(gamma=.001,C=100)

x,y = digits.data[:-10], digits.target[:-10]# everything except last value
clf.fit(x,y)
n = 3
print('Prediction',clf.predict(digits.data[[-n]]))

plt.imshow(digits.images[-n],cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
