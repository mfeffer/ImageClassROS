from siftUtils import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from random import random

## specify parameters #
k = 15                  # number of vocab features
n_nn = 7                # parameter for knn classifier
added_limit = 12        # number of features per image
hue_multi = 0           # number of times to include hue info in feature descriptor
sat_multi = 0           # number of times to include sat info in feature descriptor
random_samples = 0      # number of random patches to include is features
include_salient = True # whether to guarantee most salient feature is included

## constants ##
image_labels = ["coral","sand","floor","lego"]
descriptorLength = 128+hue_multi+sat_multi

## get feature labels ##
labelGroups = [return_labels([image_labels[i]], added_limit, hue_multi, sat_multi, random_samples, include_salient) for i in range(len(image_labels))]

## make training and testing sets ##
testGroups = []
trainingGroups = []
testGroupSize = 0.1  # take ~10% of input data to test
for i in range(len(image_labels)):
    testPictures = []
    trainingPictures = []
    for picture in labelGroups[i]:
        if random() < testGroupSize:
            testPictures.append(picture)
        else:
            trainingPictures.append(picture)
    testGroups.append(testPictures)
    trainingGroups.append(trainingPictures)

## agglomerate features ##
allTrainingLabels = np.concatenate(trainingGroups)
allTrainingFeatures = allTrainingLabels.reshape(-1, descriptorLength)

## cluster to build vocabulary ##
fcl = KMeans(n_clusters=k, random_state=1)
fcl.fit(allTrainingFeatures)

## generate term vector for each training image ##
def get_term_vector(picture):
    tv = [0 for i in range(k)]
    features = picture.reshape(-1, descriptorLength)
    for feature in features:
        result = fcl.predict([feature])
        tv[int(result)] +=1
    return tv

## get term vectors and classify ##
Xtv = []
ylabels = []
for i in range(len(image_labels)):
    for picture in labelGroups[i]:
        tv = get_term_vector(picture)
        Xtv.append(tv)
        ylabels.append(i)

tvClassifier = KNeighborsClassifier(n_neighbors=n_nn)
tvClassifier.fit(Xtv,ylabels)


### Test ###
test_successes = 0
test_failures = 0

training_successes = 0
training_failures = 0

training_results = np.zeros((4,4))
test_results = np.zeros((4,4))

for i in range(len(image_labels)):
    test_row = np.zeros(4)
    for picture in testGroups[i]:
        classification = tvClassifier.predict([get_term_vector(picture)])
        if int(classification)==i:
            test_successes +=1
        else:
            test_failures +=1
        test_row[int(classification)]+=1
    test_results[i] = test_row/ test_row.sum()
    

    training_row = np.zeros(4)   
    for picture in trainingGroups[i]:
        classification = tvClassifier.predict([get_term_vector(picture)])
        if int(classification)==i:
            training_successes +=1
        else:
            training_failures +=1
        training_row[int(classification)]+=1
    training_results[i] = training_row/ training_row.sum()

print "Test Success: " + str(float(test_successes)/(test_successes+test_failures))
print "Training Success: " + str(float(training_successes)/(training_successes+training_failures))

import matplotlib.pyplot as plt
plt.imshow(test_results, interpolation='none', cmap='Greys')
plt.show()
