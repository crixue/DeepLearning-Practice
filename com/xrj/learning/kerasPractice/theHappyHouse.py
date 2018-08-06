#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from com.xrj.learning.kerasPractice.kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import aetros.backend


# GRADED FUNCTION: HappyModel
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)  #axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=1。
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(X)

    X = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X)
    return model


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model = HappyModel(X_train.shape[1:])
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
    his = model.fit(X_train, Y_train, batch_size=32, epochs=10)  #要训练的模型-train
    path = './visualization/happy_house_train.txt'
    # job = aetros.backend.context()
    # with open(path, 'w') as f:
    #     f.write(str(his.history))
    # job.add_embedding_word2vec(x=1, path=path)

    preds = model.evaluate(X_test, Y_test)  #要评估的模型-test
     # model.predict(X_train)
    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    img_path = "./images/test_happy.jpg"
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x))

    model.summary()
    plot_model(model, to_file="./images/test_happy.jpg")
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
