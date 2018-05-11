# BOF Model

This folder contains the code for the BOF functionality. Follow the steps below to train:

1. Download training images (already annotated). These should be put in a folder named "training" in the same directory as the python scripts. Within the training folder, there should be folders for each classification, full of examples of those classes.
2. Run bof_train.py, which will produce vocabClassifier.pkl and termVectorCLassifier.pkl

Follow the steps below to classify:
1. api.py will be able to interface with ros and take in a ros image and outputs  classification.
2. This calls bof_classify which uses the term vector classifier and the feature classifier as pickled objects in the folder.
