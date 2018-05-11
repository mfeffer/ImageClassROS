from PIL import Image
import numpy as np
from keras.models import load_model
import keras as K
import tensorflow as tf
import cv2

classes = ["coral", "lego", "floor", "sand"]
DEFAULT_PIXEL_LOCATIONS = [(110, 390), (110, 250), (320, 250), (320, 390), (530, 250), (530,390), (80,80)]

#global model
model = load_model('grand_challenge_trained_new.h5')
#    global graph
graph = K.backend.get_session().graph

# Given a path to an image, returns the classification at some points in space
# Params:
#       img - np ndarray corresponding to RGB image
#       points (OPTIONAL) - list of (x,y) tuples to classify, from the bottom left corner
#                           Defaults to six points in the bottom half of the image
#       img_display (OPTIONAL) - True if pictures should be printed, defaults to False
# Returns: Tuple with (x position, y position, classification probability array)
#           Prob array is [coral, lego, floor, sand]
def classify_image(img, points=DEFAULT_PIXEL_LOCATIONS, img_save=False):

    results = []
    
    #img = img.convert("RGBA")
    patch_size = (80,80)
    for point in points:
        x, y = point
        #y = 480 - y
        box = (int(x - patch_size[0]/2), int(y - patch_size[1]/2), int(x + patch_size[0]/2), int(y + patch_size[1]/2))

        cropped = img.crop(box).resize((200,200))
        Rvals = np.array(cropped.getdata(band=0)).reshape((200,200))
        Gvals = np.array(cropped.getdata(band=1)).reshape((200,200))
        Bvals = np.array(cropped.getdata(band=2)).reshape((200,200))
        hues = np.array(cropped.convert("HSV").getdata(band=0)).reshape((200,200))

        # Normalize
        Rvals = Rvals / 255.0
        Gvals = Gvals / 255.0
        Bvals = Bvals / 255.0
        hues = hues / 360.0
        print("Rvals")
        print(Rvals)
        print("Gvals")
        print(Gvals)
        print("Bvals")
        print(Bvals)
        print(hues)

        RGBHimage = np.stack((Rvals, Gvals, Bvals, hues), axis=-1)
        expanded = np.expand_dims(RGBHimage, axis=0)
        probs = None
        with graph.as_default():
            probs = model.predict(expanded)[0]
        # result = -1
        # if probs[3] > 0.17:
        #     result = 3
        # else:
        #     result = model.predict_classes(expanded)[0]

        result = None
        with graph.as_default():
            result = model.predict_classes(expanded)[0]
        if img_save:
            

            # RED IS CORAL
            # GREEN IS LEGO
            # BLUE IS FLOOR
            # YELLOW IS SAND
            color = [(255, 0, 0, 50), (0, 255, 0, 50), (0, 0, 255, 50), (255, 255, 0, 50)]

            overlay_color = color[result]
            overlay = Image.new('RGBA', patch_size, overlay_color)
            img.paste(overlay, box=(box[0], box[1]))

        x, y = getXYFromPixel(point)
        #final_result = (x, y, probs)
        #results.append(final_result)
        results.append(x)
        results.append(y)
        results.append(result)
        print(result)
        results.extend(probs.tolist())   

    if img_save:
        img.save("/home/michael/overlay_new.png")

    return results

def getXYFromPixel(pixel):
    x_pix, y_pix = pixel

    # First get the x position from the fit
    distanceFromBottom = 6.497747897*1.004078138**y_pix + 3

    # Now get the y position
    midPixel = 640 / 2
    distanceFromCenter = abs(midPixel - x_pix)
    theta = distanceFromCenter / midPixel
    x_pos = np.tan((62.2/360) * np.pi)*theta*distanceFromBottom
    if x_pix < midPixel:
        x_pos *= -1

    return distanceFromBottom, x_pos

if __name__ == '__main__':
    path_to_image = "/home/michael/ROSClassify/new_test/imgDistance2.jpg"
    cv_img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    cv_img = Image.fromarray(cv_img[...,::-1])
    img = Image.open(path_to_image)
    np.testing.assert_array_equal(img, cv_img)
    img = img.resize((640, 480))

    classifs = classify_image(img, model, graph, img_save=True) # (mix - 31) (sand - 57, 53) (none - 58) (coral - 38) (cool result - 79)
    split_data = [classifs[i:i+7] for i in range(0, len(classifs), 7)]
    print(split_data)
