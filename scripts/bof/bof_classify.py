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

k = featureClassifier.n_clusters # number of vocab features
added_limit = 18
image_labels = ["coral","sand","floor","lego"]

PATCH_SIZE = (80,80)
DEFAULT_PIXEL_LOCATIONS = [(110, 390), (110, 250), (320, 250), (320, 390), (530, 250), (530,390), (80,80)]

def getPatches(img):
    cropped = []
    dx = PATCH_SIZE[0]/2
    dy = PATCH_SIZE[1]/2
    for y,x in DEFAULT_PIXEL_LOCATIONS:
        cropped_img = img[x-dx:x+dx, y-dy:y+dy]
        #cropped_img = cv2.resize(cropped_img, (200,200))
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
    features = get_image_descriptor(cropped_img)
    tv = get_term_vector(features)
    classification = int(tvClassifier.predict([tv]))
    return image_labels[classification]

def show(cp):
    print classify_patch(cp)
    cv2.imshow("patch",cp)
    cv2.waitKey(0)

def processImage(imgpath = "ros_images/ros_0.jpg"):
    img = cv2.imread(imgpath)
    patches = getPatches(img)
    for patch in patches:
        show(patch)


