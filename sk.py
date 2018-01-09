import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()
clf = svm.SVC(gamma=.0001,C=100)

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)
n = int(input('Which element of the data set do you want to check'))
print('Prediction',clf.predict(digits.data[[n]]))

#Display the Image on which the algorithim is run
plt.imshow(digits.images[n],cmap=plt.cm.gray_r)
plt.show()
