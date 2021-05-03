# VERSION 2

# we need to calculate the mean and standard deviation of each feature in
# each class and then the class conditional probabilities of each feature in each class
# iris data :
"""

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa -> class label 0
-- Iris Versicolour -> class label 1
-- Iris Virginica -> class label 2
"""

import math
import random
import csv


def encode_class(mydata):
    # this function encode the class label to integer so we can work well with mean and some  calculation
    # ex : [ [f1,f2,f3,c1=0] , ..... , [f1,f2,f3,c2=1] , ..... , [f1,f2,f3,c3=2] , .....] <-like this
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata


# function that take the data and split it into training set and test set by the split value
def handleDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])  # to convert the string value in all dataset to float number
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def groupUnderClass(mydata):  #

    dict = {}
    for i in range(len(mydata)):
        if mydata[i][-1] not in dict:
            dict[mydata[i][-1]] = []
        dict[mydata[i][-1]].append(mydata[i])
    return dict


# Calculating Mean
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculating Standard Deviation
def std_dev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def MeanAndStdDev(mydata):
    info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)]
    # eg: list = [ [a, b, c], [m, n, o], [x, y, z]]
    # here mean of 1st attribute =(a + m+x)/3, mean of 2nd attribute = (b + n+y)/3
    # delete summaries of last class
    del info[-1]
    return info


# find Mean and Standard Deviation under each class
def MeanAndStdDevForClass(mydata):
    info = {}
    dict = groupUnderClass(mydata)
    for classValue, instances in dict.items():
        info[classValue] = MeanAndStdDev(instances)
    return info

# Calculate Gaussian Probability Density Function
def calculateGaussianProbability(x, mean, stdev):
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo


# Calculate Class Probabilities
def calculateClassProbabilities(info, test):

    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)

    return probabilities


# Make prediction - highest probability is the prediction
def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# returns predictions for a set of examples
def getPredictions(info, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(info, test_set[i])
        predictions.append(result)
    return predictions


# Accuracy score
def accuracy_rate(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0

# main code


trainingSet = []
testSet = []
handleDataset('iris.data', 0.5, trainingSet, testSet)

trainingSet = encode_class(trainingSet)
testSet = encode_class(testSet)


print('..................')
info = MeanAndStdDevForClass(trainingSet)
print('info is: ')

for i in info:
    print(i)
    print(info[i])
print('***********')

probs=calculateClassProbabilities(info,testSet[1])
print('probs of one test pattern')
print(probs)
print('*********************************************')
print('')
print('')

predictions = getPredictions(info, testSet)
accuracy = accuracy_rate(testSet, predictions)
print("Accuracy of your model is: ", accuracy)
print('')
print('')

print('*********************************************')
print('')
print('')

nn=calculateGaussianProbability(5.1,4.983,0.3)
print('nn is : ')
print(nn)
print('')
print('')
print('')
print('')
