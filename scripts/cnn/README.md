# CNN Model

This folder contains the code for the CNN functionality. Follow the steps below to run:

1. Download a model from [this Drive folder](https://drive.google.com/drive/folders/1hmrUoPmDxhI37l57LdAsowS8hapfM1wN?usp=sharing). We recommend starting with `gc_ros_eq.h5`.
2. Change the path to the model in `api.py`
3. Run `api.py` to classify an image (change line 112 for choosing a specific image).

This will classify segments of an image at certain pixels. At these locations, it will also return the approximate x and y location in inches relative to the robot.

# Instructions To Train
1. Download the [raw images](https://drive.google.com/open?id=1cjuvRTpggDX2W_G4E-m1HGAAvhdamkRq) and [training images](https://drive.google.com/open?id=1CD3ccvi3KJQEOYaqbz8TJHP4_NYZ3N1g) from Google Drive **Note, only download the raw images if you are running step 3)**. *New training images are located [here](https://drive.google.com/file/d/1gdhJ4HYloUfxv_Tigw-a3IA5tXU1aWqS/view?usp=sharing).*
2. Extract these folders to the root folder of the project (i.e. you should now have two new folders, `raw` and `training`)
3. Run `python make_training.py` to load image segments and classify (NOTE: Images not auto increment! Just make sure you download the most recent from step 1, and upload new versions when done creating new training images)
4. Run `python train.py` to train the model on lego vs coral vs floor. Note that this only uses 91 images from each class, so you can change.
5. Run `python3 api.py` to classify segments from an image. **Make sure to change the path to the model that you want to use.**