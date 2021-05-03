import csv
import random
import math
import operator


# function that calculate the ecludien Distance
def ecludienDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


# function that calculate the Mean of a set of points
def CalculateMean(setofdata):
    MeanPattern = []
    total_of_setofdata = len(setofdata)
    length_of_each_point_in_set = len(setofdata[0]) - 1  # -1 because without the label value of each pattern in dataset
    # print(total_of_setofdata)
    # print(length_of_each_point_in_set)s
    sum = 0
    for i in range(length_of_each_point_in_set):
        for j in range(total_of_setofdata):
            sum += setofdata[j][i]
            res = sum / total_of_setofdata
        MeanPattern.append(res)

        # Reset the sum
        sum = 0

    # print('The mean value is : ')
    # print(MeanPattern)

    return MeanPattern




# function thst take the data and split it into training set and test set by the split value
def handleDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])  # to convert the string value in all dataset to float number
            if (random.random() < split):
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# function that take the training set as input and return for each class it's mean
def MeansForEachClass(trainingSet):
    eachClass = []

    means = []
    means_dic = {}
    tempSet = []
    for i in range(len(trainingSet)):
        theclassLabel = trainingSet[i][-1]
        # print(theclassLabel)
        if theclassLabel not in eachClass:
            eachClass.append(theclassLabel)
    number_of_Class = len(eachClass)

    for i in range(number_of_Class):
        for j in range(len(trainingSet)):
            if trainingSet[j][-1] == eachClass[i]:
                tempSet.append(trainingSet[j])
        #print(tempSet)
        mean_of_temp_set = CalculateMean(tempSet)
        means.append(mean_of_temp_set)
        means_dic[eachClass[i]] = mean_of_temp_set
        tempSet.clear()
    return means_dic;


# function that take the training set and a test pattern and predict the class of the test pattern
def predictClassMDC(trainingSet, testPattern):
    distances = {}
    means_of_trainiset = MeansForEachClass(trainingSet)
    for x in means_of_trainiset:
        distance = ecludienDistance(means_of_trainiset[x], testPattern, 3)
        distances[x]=distance

    #print(distances)

    sortedclassesbasedondistance = sorted(distances.items(), key=operator.itemgetter(0))

    #print(sortedclassesbasedondistance[0][0])
    PredictedClass=sortedclassesbasedondistance[0][0]

    return PredictedClass



#function that get accurency
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        realClass=testSet[x][-1]
        predictedClass = predictions[x]
        if realClass == predictedClass:
            correct=correct+1

    return  (correct / float(len(testSet)))*100.0



def main():
    trainingSet = []
    testSet = []
    split = 0.5

    handleDataset('iris.data', split, trainingSet, testSet)
    print('Train set : ' + repr(len(trainingSet)))
    print('Test set : ' + repr(len(testSet)))

    predictions = []
    for x in range(len(testSet)):
        predictedClass=predictClassMDC(trainingSet,testSet[x])
        predictions.append(predictedClass)
        #print(predictedClass)

    accurency=getAccuracy(testSet,predictions)
    print('Accurency:' + repr(accurency) + '%')


main()

#testData


trainSet = [[2, 2, 2, 'a'], [1.5, 4, 3, 'a'], [4, 6, 1, 'c'], [1, 2, 3, 'a'], [4, 4, 4, 'b'], [3, 3, 3, 'b'],
            [4.5, 4.55, 4.5, 'b']]
CalculateMean(trainSet)
trainSet = [[1, 6, 2, 'b'], [4, 6, 1, 'b'], [2, 3, 4, 'a'], [4, 7, 1, 'a'], [2, 2, 2, 'a'], [1.5, 4, 3, 'a'],
            [4, 6, 1, 'c'], [1, 2, 3, 'a'], [4, 4, 4, 'b'], [3, 3, 3, 'b'], [4.5, 4.55, 4.5, 'b']]
print('  Traininf set is : ')
print(trainSet)
print('*********')
# MeansForEachClass(trainSet)
testP = [10, 0, 10, 'c']
predictClassMDC(trainSet, testP)
