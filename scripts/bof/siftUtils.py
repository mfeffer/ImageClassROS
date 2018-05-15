import numpy as np
import cv2
import os
from SIFTDescriptor import SIFTDescriptor
from scipy.ndimage import filters


'''
    return the image descriptor for all images in a set of directories
    params:
        directoryList: list of directories to be skimmed
        added_limit: how many features are picked out of each image
        hue_multi: how many times the patch hue is added to the patch descriptor
        sat_multi: how many times the patch saturation is added to the patch descriptor
        random_samples: how many random patches are included as features
        include_salient: whether the most salient patch is included as a feature
        patch_size: size of feature to be described
    returns:
        a 2d array where each row is an image, and each image is a concanteation of all features
        for each image, there are added_limit+random_samples+include_salient features, and each feature is given a descriptor of length (128+hue_multi+sat_multi)
'''
def return_labels(directoryList, added_limit, hue_multi=0, sat_multi=0, random_samples=0, include_salient=False, patch_size=32):
    nFeatures = added_limit+random_samples+include_salient
    range_size = nFeatures*(128+hue_multi+sat_multi)
    count_limit = 100
    print(added_limit)
    print("range: " , range_size)
    print("count: " , count_limit)
    SD = SIFTDescriptor(patchSize = patch_size)

    #creates initial vector of size 3968
    sift_pictures = np.asarray([[0 for _ in range(range_size)]])
    for directory in directoryList:
        #go through the directories with the relevant images
        for filename in sorted(os.listdir("training/"+directory)):
            name = "training/"+directory + '/' + filename
            if name[-3:] == 'png' or name[-3:] == 'jpg':
                image = cv2.imread(name)
                image_sift_features = get_image_descriptor(image, added_limit, hue_multi, sat_multi, random_samples, include_salient, patch_size)
                if len(image_sift_features)==range_size:
                    sift_pictures = np.append(sift_pictures, [image_sift_features], axis=0)

    return sift_pictures[1:]

'''  return the average hue of a patch  '''
def getHue(colorPatch):
    hsv = cv2.cvtColor(colorPatch, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,0])

'''  return the average hue of a patch  '''
def getSat(colorPatch):
    hsv = cv2.cvtColor(colorPatch, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,1])


'''
    return the image descriptor for a single image
    params:
        image: BGR image
        added_limit: how many features are picked out of each image
        hue_multi: how many times the patch hue is added to the patch descriptor
        sat_multi: how many times the patch saturation is added to the patch descriptor
        random_samples: how many random patches are included as features
        include_salient: whether the most salient patch is included as a feature
        patch_size: size of feature to be described
    returns:
        a concanteation of all feature descriptions for a single image
        for each image, there are added_limit+random_samples+include_salient features, and each feature is given a descriptor of length (128+hue_multi+sat_multi)
'''
def get_image_descriptor(image, added_limit, hue_multi=0, sat_multi=0, random_samples=0, include_salient=False, patch_size=32):
    nFeatures = added_limit+random_samples+include_salient
    range_size = (nFeatures)*(128+hue_multi+sat_multi) #total length of image descriptor = # features * len(feature description)
    count_limit = 50
    SD = SIFTDescriptor(patchSize = patch_size)
       
    #transform it to black and white and get features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = cv2.goodFeaturesToTrack(gray,count_limit,0.01,10)
 
    features = getRandomSamples(random_samples, image, patch_size)
    if include_salient:
        features = np.concatenate((getSalientPoint(gray, patch_size), features))
    features = np.concatenate((features, corners))
    features = np.int0(features)
    
    image_sift_features = np.asarray([]) 
    count, actually_added = 0, 0
        
    #Here we want to get added_limit images for our feature vector and then break out of the loop
    while actually_added < nFeatures and count < count_limit and count<len(features):
        feature = features[count][0]
        
        if patch_size/2 <= feature[1] <= h-patch_size/2 and patch_size/2 <= feature[0] <= w-patch_size/2:
            patch = gray[feature[1]-int(patch_size/2):feature[1]+int(patch_size/2), feature[0]-int(patch_size/2):feature[0]+int(patch_size/2)]
            colorPatch = image[feature[1]-int(patch_size/2):feature[1]+int(patch_size/2), feature[0]-int(patch_size/2):feature[0]+int(patch_size/2)]
            hue = getHue(colorPatch)
            sat = getSat(colorPatch)
            sift = SD.describe(patch)
            sift = np.append(sift, np.array([hue]*hue_multi+[sat]*sat_multi))
            image_sift_features = np.append(image_sift_features, sift)
            actually_added += 1
        count += 1
    return image_sift_features

def getRandomSamples(n, image, patch_size):
    h,w,_ = image.shape
    minx = patch_size/2
    maxx = w - patch_size/2
    miny = patch_size/2
    maxy = h - patch_size/2
    samples = np.zeros((n,1,2))
    np.random.seed(0)
    for i in range(n):
        x = np.random.randint(minx,maxx)
        y = np.random.randint(miny,maxy)
        samples[i] = [x,y]

    return samples

def getSalientPoint(gray, patch_size):
    hist = np.histogram(gray, 256, (0,255))[0]
    hist = hist/float(sum(hist))
    logp = np.log2(hist, where=(hist>0))
    toLogP = lambda x: -logp[x]
    toLogP = np.vectorize(toLogP)
    bitSal = toLogP(gray)
    patch = np.ones((patch_size,patch_size))
    patchSal = filters.convolve(bitSal, patch)
    sp = np.zeros((1,1,2))
    sp[0] = np.array(np.unravel_index(np.argmax(patchSal, axis=None), patchSal.shape))
    return sp
    

    

    
    
