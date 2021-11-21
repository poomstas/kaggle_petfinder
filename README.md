# PetFinder's Kaggle Competition

More information [here](https://www.kaggle.com/c/petfinder-pawpularity-score/overview).


## Training, Validation, Testing Data Structure
The original dataset provides summarized .csv files for training and testing. In this work, I separate the training data into training and validation.

The helper function `separate_train_val` is used to separate the the training data into training and validation, and store that information in the ./ data folder.

The final resulting folder is as follows:

~~~
