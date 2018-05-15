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

## agglomerate features ##
allTrainingLabels = np.concatenate(labelGroups)
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

## save classifiers ##
out = open("vocabClassifier.pkl", "w")
pickle.dump(featureClassifier, out)
out.close()

out = open("termVectorClassifier.pkl","w")
pickle.dump(tvClassifier, out)
out.close()
