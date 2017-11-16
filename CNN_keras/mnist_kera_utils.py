import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def digit_rec_model(input_shape):

    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(8, (4, 4), strides = (1, 1), padding = 'same', name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((8, 8), strides = (8, 8), padding = 'same', name = 'max_pool_1')(X)
    X = Conv2D(16, (2, 2), strides = (1, 1), padding = 'same', name = 'conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), strides = (4, 4), padding = 'same', name = 'max_pool_2')(X)
    X = Flatten()(X)
    X = Dense(10, activation = 'softmax', name = 'fc')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'digit_rec_model')

    return model