# image-classification-cifar-10-kaggle-competition

CIFER-10 kaggle competition data link:
https://www.kaggle.com/competitions/cifar-10/data

- Model: Custom Deep convolutional network model with 4 convolutional blocks containing Conv2d layers, BatchNorm2d layers, ReLU layers, MaxPool2d and Dropout used as sequential block.
- Optimization: Used Adam optimizer along with weight decay and Learning Rate schedular.
- Data Preprocessing: Transformed 32x32 image to 64x64 image for better training and inference. Used 10% of the train folder data as validation data.
- Performance: Could achieve 93.5% accuracy on training and 88.2% accuracy during validation