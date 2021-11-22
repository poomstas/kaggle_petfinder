# PetFinder's Kaggle Competition

More information [here](https://www.kaggle.com/c/petfinder-pawpularity-score/overview).

## Core Challenge
The challenge is to generate a regression model that predicts the score representing the popularity of the pet images (aka "Pawpularity"). What's a little more interesting is that the challenge also provides boolean metadata (see table below) that could help improve the regression accuracy. We will have to devise a way to incorporate both the image and the metadata to predict a continuous, numerical popularity value.

![Training Data](https://github.com/poomstas/kaggle_petfinder/blob/dev/markdown/training_data.png "Training Data")

## Novelty
- Use of embedding layers to...

## Training, Validation, Testing Data Structure
The original dataset provides summarized .csv files for training and testing. In this work, I separate the training data into training and validation.

The helper function `separate_train_val` is used to separate the training data into training and validation, and store that information in the ./ data folder.

The final resulting folder is as follows:
