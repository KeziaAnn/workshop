# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
iris=load_iris()
features=iris.data
labels=iris.target
# instantiate learning model (k = 3)
knn = KNeighborsClassifier()
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=.3)


# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

