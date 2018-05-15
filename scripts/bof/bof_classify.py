from siftUtils import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pickle


inp = open("vocabClassifier.pkl", "r")
featureClassifier = pickle.load(inp)
inp.close()

inp = open("termVectorClassifier.pkl", "r")
tvClassifier = pickle.load(inp)
inp.close()


## specify parameters #
k = featureClassifier.n_clusters    # number of vocab features
n_nn = 7                            # parameter for knn classifier
added_limit = 12                    # number of features per image
hue_multi = 0                       # number of times to include hue info in feature descriptor
sat_multi = 0                       # number of times to include sat info in feature descriptor
random_samples = 0                  # number of random patches to include is features
include_salient = True              # whether to guarantee most salient feature is included

image_labels = ["coral","sand","floor","lego"]

PATCH_SIZE = (80,80)
DEFAULT_PIXEL_LOCATIONS = [(110, 390), (110, 250), (320, 250), (320, 390), (530, 250), (530,390), (80,80)]

## return a list of PATCH_SIZE image slices from the input image, taken from DEFAULT_PIXEL_LOCATIONS ##
def getPatches(img):
    cropped = []
    dx = PATCH_SIZE[0]/2
    dy = PATCH_SIZE[1]/2
    for y,x in DEFAULT_PIXEL_LOCATIONS:
        cropped_img = img[x-dx:x+dx, y-dy:y+dy]
        cropped.append(cropped_img)
    return cropped
    
## generate term vector for each training image ##
def get_term_vector(picture):
    tv = [0 for i in range(k)]
    features = picture.reshape(-1, 128)
    for feature in features:
        result = featureClassifier.predict([feature])
        tv[int(result)] +=1
    return tv

## Given a small patch as an image, return a classification from ["coral", "sand", "floor", "lego"] ##
def classify_patch(cropped_img):
    features = get_image_descriptor(cropped_img, added_limit, hue_multi, sat_multi, random_samples, include_salient)
    tv = get_term_vector(features)
    classification = int(tvClassifier.predict([tv]))
    return image_labels[classification]

## print the classification of the patch and visualize the classified patch, for validation purposes ##
def show(cp):
    print classify_patch(cp)
    cv2.imshow("patch",cp)
    cv2.waitKey(0)


## given the path to an image, return a list of pixel locations and the classification of the object at that location ##
def processImage(imgpath = "ros_images/ros_11.jpg"):
    img = cv2.imread(imgpath)
    patches = getPatches(img)
    #for patch in patches:
    #    show(patch)
    output = []
    for i in range(len(DEFAULT_PIXEL_LOCATIONS)):
        output.append([DEFAULT_PIXEL_LOCATIONS[i], classify_patch(patches[i])])
    return output

