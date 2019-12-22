import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd

# Import data
data_frame = pd.read_csv("voices.csv")
# Extract features
data_x = data_frame.values[:, 0:-1]
# Extract labels as ones and zeros
data_y =pd.Categorical( data_frame.values[:, -1]).codes

# Get number of data points
num_data = data_frame.shape[0]

# Randomize data
np.random.seed(seed=0)  # fix seed to get same sequence
idx = np.random.permutation(num_data)

# Define test, eval, train proportions
proportion_train = 0.7
proportion_eval = 0.2
proportion_test = 1 - proportion_train - proportion_eval

# Extract train, eval, and test datasets
idx_train = idx[0: int(proportion_train*num_data)]
X_train = data_x[idx_train, 0:13]
Y_train = data_y[idx_train]

idx_eval = idx[int(proportion_train*num_data): len(idx_train) + int(proportion_eval*num_data)]
X_eval = data_x[idx_eval, 0:13]
Y_eval = data_y[idx_eval]

idx_test = idx[len(idx_train) + len(idx_eval):]
X_test = data_x[idx_test, 0:13]
Y_test = data_y[idx_test]




print("Done!")