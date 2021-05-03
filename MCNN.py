import csv
import random
import math
import operator


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
    sum = 0
    for i in range(length_of_each_point_in_set):
        for j in range(total_of_setofdata):
            sum += setofdata[j][i]
            res = sum / total_of_setofdata
        MeanPattern.append(res)
        sum = 0

    return MeanPattern


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
        # print(tempSet)
        mean_of_temp_set = CalculateMean(tempSet)
        means.append(mean_of_temp_set)
        means_dic[eachClass[i]] = mean_of_temp_set
        tempSet.clear()

    # print(means_dic)
    return means_dic


def CreateCendensetSet(trainingSet):
    CS = []
    MeansOfClasses = MeansForEachClass(trainingSet)
    MeansPatterns = []  # List of means
    array_of_classes = []
    # print(MeansOfClasses)
    for x, y in MeansOfClasses.items():
        # print(x, y)
        array_of_classes.append(x)
        y.append(x)
        MeansPatterns.append(y)

    #print(MeansPatterns)
    #print(array_of_classes)
    # Now we want to get the neasrest pattern to the mean in each class:
    dists = []
    nearest_points_each_class = []
    nbcoordinates = len(trainingSet[0]) - 1
    #print(nbcoordinates)
    for i in range(len(MeansPatterns)):
        for j in range(len(trainingSet)):
            if trainingSet[j][-1] == MeansPatterns[i][-1]:
                d = ecludienDistance(MeansPatterns[i], trainingSet[j], nbcoordinates)
                dists.append((trainingSet[j], d))
        dists.sort(key=operator.itemgetter(1))
        res = dists[0]
        nearest_points_each_class.append(res)
        dists.clear()

    #print(nearest_points_each_class)

    # so now i have the neareset point in each class to it's mean
    # add them to the condenset set:
    for i in range(len(nearest_points_each_class)):
        CS.append(nearest_points_each_class[i][0])

    #print('')
    #print('CS is : ')
    #print(CS)

    # NOW remove the points added to CS from trainig set:
    NewTrainingSet = []
    for i in range(len(trainingSet)):
        if trainingSet[i] not in CS:
            NewTrainingSet.append(trainingSet[i])

    #print(NewTrainingSet)

    # NOW the cs containe nearest point to the mean in each class
    dis = []

    for i in range(len(NewTrainingSet)):
        for j in range(len(CS)):
            d = ecludienDistance(NewTrainingSet[i], CS[j], nbcoordinates)
            dis.append((NewTrainingSet[i], d))
        dis.sort(key=operator.itemgetter(1))
        neares_point_tocs = dis[0][0]
        if neares_point_tocs[-1] != NewTrainingSet[i][-1]:
            CS.append(neares_point_tocs)
        dis.clear()

    #print('New CS')
    #print(CS)

    return CS


def predictClass(trainingSet , TestPattern):

    CS = CreateCendensetSet(trainingSet)
    print('CS is : ')
    print(CS)
    print('')
    print('')
    nbcoordinates=len(trainingSet[0])-1
    dis=[]
    for i in range(len(CS)):
        d=ecludienDistance(CS[i],TestPattern,nbcoordinates)
        dis.append((CS[i],d))

    dis.sort(key=operator.itemgetter(1))
    print(dis)
    print(dis[0][0][-1])
    return dis[0][0][-1]

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
        predictedClass=predictClass(trainingSet,testSet[x])
        predictions.append(predictedClass)
        #print(predictedClass)

    accurency=getAccuracy(testSet,predictions)
    print('Accurency:' + repr(accurency) + '%')


main()







# Test functions:
"""
trainSet = [[1, 6, 2, 'b'], [4, 1, 8, 'c'], [5, 9, 10, 'a'], [3, 2, 5, 'b'], [5, 1, 3, 'a'], [4, 6, 1, 'b'],
            [2, 3, 4, 'a'], [4, 7, 1, 'a'], [2, 2, 2, 'a'], [1.5, 4, 3, 'a'],
            [4, 6, 1, 'c'], [1, 2, 3, 'a'], [4, 4, 4, 'b'], [3, 3, 3, 'b'], [4.5, 4.55, 4.5, 'b'], [1, 1, 1, 'd']]
print('  Training set is : ')
print(trainSet)
print('*********')

p=[1,2,3]
print('p:')
print(p)
print('')
predictClass(trainSet,p)
"""