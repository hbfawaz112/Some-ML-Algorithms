import csv
import random
import math
import operator

# Hadnling the data using csv to read from the data file .

with open(r'iris.data') as csvfile:
    lines = csv.reader(csvfile)
    print('********** All the data in my file ***********')

    for row in lines:
        print(', ' .join(row))
    print('*********************')

# A function that split the data into to categories : training set and testing set.

def handleDataset(filename , split , trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if(random.random() < split):
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
#test the function

trainingSet=[]
testSet=[]
handleDataset(r'iris.data' , 0.75,trainingSet,testSet)
print('*********** Test the splited data : training set and testSet of my data ***************')
print('Train : ' + repr(len(trainingSet)))
print('Test : ' + repr(len(testSet)))
print('*******************')


# A function the get the ecludien distance of 2 instances
   # SQRT( (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2 )
def ecludienDistance(instance1 ,instance2 , length):
    distance=0
    for i in range(length):
         distance += pow( (instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = ecludienDistance(data1 ,data2 , 3)
print('************** test the distance ecludeint of a 2 givent point(instance) **************')
print('Distance : ' + repr(distance))
print('***********************')


# Find the K nearest points
def getKNeighbors(trainingSet , testInstance , k):
    distance=[]
    length=len(testInstance)-1 # to get just the 3 x,y,z of the testance without the 4'th column wich is the label of class
    for x in range(len(trainingSet)):
        dist=ecludienDistance(testInstance,trainingSet[x], length)
        distance.append((trainingSet[x] , dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0]) # each element of distance list is in form ( instance:[4,4,4] , distance:1.23)
    return neighbors

    return neighbors

trainSet = [ [2 ,2 ,2 , 'a'] , [4, 4, 4, 'b'] , [3,3,3,'b']]
testInstance = [5,5,5] # test instance y3ne l'P new pattern
k=2
neighbors = getKNeighbors(trainSet , testInstance , k)
print("********* test a fucntion the get me the  k neigrest point to w new pattern (testInstance P) ********")
print(neighbors)
print('****************************')


#Predict the class :
#to predict the class of the point given its k closest neighbors. Let’s create
# a predictClass function to calculate the votes of each class from the k nearest neighbors.

def predictClass(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if(response in classVotes):
            classVotes[response] +=1
        else:
            classVotes[response]=1
    sortedVotes = sorted(classVotes.items() , key=operator.itemgetter(1) , reverse=True)
    return sortedVotes[0][0]


neighbors_test=[ [1,1,1,'a'] , [2,2,2,'b'] , [3,3,3,'b'] ]
print('****************TEST THE prdict class of k th neighbors**********')
print( 'the predict class of this data set  is : ' + predictClass(neighbors_test))
print('************************************')


#check the accuracy
def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        realClass=testSet[x][-1]
        predictedClass = predictions[x]
        if realClass == predictedClass:
            correct=correct+1

    return  (correct / float(len(testSet)))*100.0



testSet = [ [1,1,1,'a'] , [2,2,2,'b'] , [3,3,3,'b'] ]
predictions = ['a' , 'a' , 'a']
accuracy = getAccuracy(testSet , predictions)
print('*******************Prediction of these data is : ****************')
print(accuracy)
print('***************************')

def main():
    trainingSet=[]
    testSet=[]
    split=0.5

    handleDataset('iris.data' , split , trainingSet , testSet)
    print('Train set : ' + repr(len(trainingSet)))
    print('Test set : ' + repr(len(testSet)))

    predictions=[]
    k=3

    for x in range(len(testSet)):
        neighbors = getKNeighbors(trainingSet , testSet[x] , k)
        result = predictClass(neighbors)
        predictions.append(result)
        print(' > predicted = ' + repr(result) + ', actual = ' + repr(testSet[x][-1]))


    accuracy = getAccuracy(testSet , predictions)

    print('Accurency : ' + repr(accuracy) + '%')

print('****************Main********************')
main()
