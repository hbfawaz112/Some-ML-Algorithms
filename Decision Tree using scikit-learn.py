from sklearn import datasets, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,test_size=0.5, random_state=109)  # 50% training and 50% test


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy Decision Tree Algorithm:", metrics.accuracy_score(y_test, y_pred))
# To extract the imageof the tree
tree.plot_tree(clf)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
tree.plot_tree(clf,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.show()
fig.savefig('imagename.png')
