import csv
import random
import math
import operator


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


# function that calculate the ecludien Distance
def ecludienDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


# function that get  the K'th nearest neighbors and assign for each K neighbor and weight based on the distance
def getKNeighbors_WithWieght(trainingSet, testInstance, k):
    # this function return a list :
    #  [  [x,y,z,label,weight] , [x,y,z,label,weight] , [x,y,z,label,weight] ]
    distance = []
    distance_value = []
    length = len(
        testInstance) - 1  # to get just the 3 x,y,z of the testance without the 4'th column wich is the label of class
    for x in range(len(trainingSet)):
        dist = ecludienDistance(testInstance, trainingSet[x], length)
        distance_value.append(dist)
        distance.append((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    distance_value.sort()
    neighbors = []
    weights = []
    d_k = distance_value[k - 1]
    d_1 = distance_value[0]

    for x in range(k):
        neighbors.append(distance[x][0])  # each element of distance list is in form ( instance:[4,4,4] , distance:1.23)

    for x in range(k):
        w = (d_k - distance_value[x]) / (d_k - d_1)
        weights.append(w)

    for x in range(k):
        neighbors[x].append(weights[x])

    return neighbors


def predictClass(neighbors):
    arr = []
    for x in range(len(neighbors)):
        thelabel = neighbors[x][-2]
        thewieight = neighbors[x][-1]
        arr.append([thelabel, thewieight])

    print(arr)

    dict = {}

    for i in range(len(arr)):
        label = arr[i][0]
        w = arr[i][1]
        if (label in dict):
            dict[label] += w
            s = dict[label]
            dict[label] = s

        else:
            dict[label] = w

    print(dict)
    # Getting first key in dictionary
    res = list(dict.keys())[0]
    print('The final class is ' + str(res))
    return res



#check the accuracy
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        realClass=testSet[x][-1]
        predictedClass = predictions[x]
        if realClass == predictedClass:
            correct=correct+1

    return  (correct / float(len(testSet)))*100.0





def main():
    trainingSet=[]
    testSet=[]
    split=0.66
    handleDataset('iris.data' , split , trainingSet , testSet)



    prediction = []
    k=3
    for x in range(len(testSet)):
        neighborswithWeight=getKNeighbors_WithWieght(trainingSet , testSet[x],k)
        result = predictClass(neighborswithWeight)
        prediction.append(result)
        #print(' > predicted = ' + repr(result) + ', actual = ' + repr(testSet[x][-1]))

    accuracy = getAccuracy(testSet, prediction)

    print('Accurency : ' + repr(accuracy) + '%')



main()


trainSet = [[1, 6, 2, 'b'], [4, 6, 1, 'b'], [2, 3, 4, 'a'], [4, 7, 1, 'a'], [2, 2, 2, 'a'], [1.5, 4, 3, 'a'],
            [4, 6, 1, 'c'], [1, 2, 3, 'a'], [4, 4, 4, 'b'], [3, 3, 3, 'b'], [4.5, 4.55, 4.5, 'b']]
testInstance = [1, 1, 1]  # test instance y3ne l'P new pattern
k = 5
neighbors = getKNeighbors_WithWieght(trainSet, testInstance, k)
print('   ')

print("********* test a fucntion the get me the  k neigrest point to w new pattern (testInstance P) ********")
print('   ')

print(neighbors)
print('   ')

print('****************************')

print('   ')
predictClass(neighbors)

