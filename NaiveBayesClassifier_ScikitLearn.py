from sklearn import datasets, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, MultinomialNB

# Load dataset
iris = load_iris()

### Splitting Data
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5,random_state=109)  # 50% training and 50% test

# Create a Gaussian Classifier (Suppose a gussian distribution of features)
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy Gaussian Naive Bayes Algorithm:", metrics.accuracy_score(y_test, y_pred)*100)

#######################################################
# Create a categorical Classifier (traditional method)
cat = CategoricalNB()
# Train the model using the training sets
cat.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = cat.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy Categorical Naive Bayes Algorithm:", metrics.accuracy_score(y_test, y_pred)*100)

#########################################################
# Create a Bernoulli Classifier (Suppose a bernoulli distribution of features)
br = BernoulliNB()
# Train the model using the training sets
br.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = br.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy Bernoulli Naive Bayes Algorithm:", metrics.accuracy_score(y_test, y_pred)*100)

##########################################################
# Create a Multinomial Classifier (Suppose a multinomial distributed data)
mn = MultinomialNB()
# Train the model using the training sets
mn.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = mn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy Multinomial Naive Bayes Algorithm:", metrics.accuracy_score(y_test, y_pred)*100)
