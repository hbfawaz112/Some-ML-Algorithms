# Version 1 :
import csv
import random
import operator


# function that take the data and split it into training set and test set by the split value
def handleDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                # to convert the string value in all dataset to float number
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# function that return array of classes : [c1,c2,c3]
def arrayofclasses(trainigset):
    # return [c1,c2,c3]
    arr = []
    for i in range(len(trainigset)):
        if trainigset[i][-1] not in arr:
            arr.append(trainigset[i][-1])
    # print(arr)
    return arr


# function return foreach class how many pattern there are inside of it
# ex : {Class 1 : 20 , Class 2: 30 , Class3 : 40}
def nbofpatternineachclass(trainingSet):
    a = 1
    arr = arrayofclasses(trainingSet) #[c1,c2,c3]
    c = 0
    res = {}
    for i in range(len(arr)):#loop of c1,c2,c3
        for j in range(len(trainingSet)):
            res[arr[i]] = 0
            if trainingSet[j][-1] == arr[i]:
                c = c + 1
        res[arr[i]] = c
        c = 0
    # print(res)
    return res#{Class 1 : 20 , Class 2: 30 , Class3 : 40}


# function that take a training set and
# assign for each class his prioir probability  P(ci) of each class :
# p(c1)=? / p(c2)=? / .....
def CalculatePriorProb(trainingSet):  # return { c1:p(c1) , c2: p(c2), ... }
    totalnbtraining = len(trainingSet)
    nb_of_patternin_each_class=nbofpatternineachclass(trainingSet)
    res={}
    for key,val in nb_of_patternin_each_class.items():
        val=val/totalnbtraining
        res[key]=val

    #print(res)
    return res


def nb_of_each_features_in_each_class_of_test_pattern(trainingSet, testPattern):
    # return {'Iris-setosa': [1, 3, 7, 12], 'Iris-versicolor': [0, 3, 0, 0], 'Iris-virginica': [0, 4, 0,
    # 0]} c1: [nbofeachfeuture in each class of the test pattern] 'Iris-setosa': [1, 3, 7, 12] -> 1 is the number of
    # testpattern[0] in class 1 , 3 is the number of testpattern[1] in class 1
    # print(trainingSet)
    # print(testPattern)
    dict = {}
    classes = arrayofclasses(trainingSet) #[c1,c2,c3]
    for i in range(len(classes)):
        dict[classes[i]] = []

    #{c1:[],c2:[],c3:[]}

    c = 0
    for i in range(len(classes)):
        for j in range(len(testPattern) - 1):
            for k in range(len(trainingSet)):
                if trainingSet[k][-1] == classes[i] and trainingSet[k][j] == testPattern[j]:
                    c = c + 1
            dict[classes[i]].append(c)
            c = 0
    # print(dict)
    return dict


# fct take train set and test pattern t , 23ml calculate la
# p(c_i/t(f1,f2,f3,f4)) = 1/p(f1,f2,f3,f4)*p(c_i)*( p(f1/c_i) * p(f2/c_i) * p(f3/c_i) * p(f4/c_i) )
def claculeClassConditionalProbs(trainingSet, testPattern):
    # p(f1,f2,f3,f4/c1) = multplication of p(fi/c1) = nboff1/totalnbofc1 * .....

    # return {c1:p(f1,f2,f3,f4/c1) ,c3:p(f1,f2,f3,f4/c1) ,c3:p(f1,f2,f3,f4/c3) , }
    dict = {}
    nb_of_pattern_each_class = nbofpatternineachclass(trainingSet)
    # print(nb_of_pattern_each_class)
    #print('')
    nb_of_feuture_of_test_in_train = nb_of_each_features_in_each_class_of_test_pattern(trainingSet, testPattern)
    # print(nb_of_feuture_of_test_in_train)
    #print('')
    array_of_classes = arrayofclasses(trainingSet)
    # print(array_of_classes)
    #print('')
    m = 1
    for i in range(len(array_of_classes)):
        total_nb_in_class = nb_of_pattern_each_class[array_of_classes[i]]
        # print(total_nb_in_class)
        for key, val in nb_of_feuture_of_test_in_train.items():
            if key == array_of_classes[i]:
                for x in range(len(val)):
                    m = m * (val[x] / total_nb_in_class)
        dict[array_of_classes[i]] = m
        m = 1

    # print(dict)
    return dict


# fct calculate the posteriori probabilities of a test pattern in the training set
def calculate_posteriori_probs(train, test):

    dict = {}
    priro_probs = CalculatePriorProb(train) # {c1:p(c1) , ...}
    #print(priro_probs)
    class_conditional_prob = claculeClassConditionalProbs(train, test) # {c1:p(p/c1) , .....}
    #print(class_conditional_prob)

    for key, val in priro_probs.items():
        for key1, val1 in class_conditional_prob.items():
            if key == key1:
                m = val * val1
        dict[key] = m
    #print(dict)
    return dict
    ##{c!:p(c1/p) = p(p/c1_*p(c1)/p(p1)}




def predict(train, test):
    posteriori_probs = calculate_posteriori_probs(train, test)
    predicted_class_label = max(posteriori_probs.items(), key=operator.itemgetter(1))[0]
    #print(predicted_class_label)
    return predicted_class_label#ex: return c1


def predict_test_set(trainingSet, testSet):
    predictions = []
    for i in range(len(testSet)):
        predicted_class = predict(trainingSet, testSet[i])
        predictions.append(predicted_class)

    #print(predictions)
    return predictions
#[c1,c2,c3,c1]


# Accuracy score
def accuracy_rate(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0

########################################################################################################################
train = []
test = []
handleDataset('iris.data', 0.5, train, test)
"""
arrayofclasses(train)
CalculatePriorProb(train)
nbofpatternineachclass(train)
nb_of_each_features_in_each_class_of_test_pattern(train, test[1])
claculeClassConditionalProbs(train, test[0])"""
# calculate_posteriori_probs(train, test[0])
#print(test)
predict(train,test[len(test)-4])
predictionss = predict_test_set(train, test)
print('the accurency of the classification predicition is : ')
print(accuracy_rate(test, predictionss))

#CalculatePriorProb(train)