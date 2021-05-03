from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from  sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
iris = load_iris()

print("Feature Names : " + str(iris.feature_names))

print("Class names : " + str(iris.target_names))

X = iris.data

y= iris.target

X_train , X_test , y_train , y_test = train_test_split( X,y , test_size=0.33, shuffle=True , random_state=4)

K_range = range(1,26)
scores={}
scores_list=[]

for k in K_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train , y_train)
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test , y_pred)
    scores_list.append(metrics.accuracy_score(y_test , y_pred))

print(scores)



plt.plot(K_range , scores_list)

plt.xlabel("Value pf K from KNN")
plt.ylabel("Testing Accuracy")

plt.show()