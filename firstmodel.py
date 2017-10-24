from sklearn.tree import DecisionTreeClassifier

features =[[140,0],[130,0],[150,1],[170,1]]
labels =['apple','apple','orange','orange']

clf =DecisionTreeClassifier()
clf.fit(features,labels)

p=clf.predict([[160,1]])
print("Prediction =",p)
