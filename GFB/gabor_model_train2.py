"""
File: gabor_model_train2.py
Call: gabor_model_train2.py <train_data_str> <test_fold> <l2> <output model>
Auth: C. DiMattina
Desc: This program defines and trains a simple binary classifier operating
      on the outputs of a bank of Gabor filters defined at multiple
      spatial scales
"""
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

from dataset_KFold import KFoldDataset
from load_dataset import LoadDataset
from Scripts.prep_dataset import PrepDataset
from gabor_class_model import GaborClass

# Set random seeds for reproducible results --- only works in CPU environment
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

learning_rate       = 0.001
batch_size          = 500
num_epochs          = 10
num_folds           = 5
pool_size           = (8,8)
strides             = (8,8)

train_data_in       = sys.argv[1]
test_fold           = int(sys.argv[2])
l2                  = pow(10.0,int(sys.argv[3]))
save_dir            = sys.argv[4]

if train_data_in== "EDGE":
    dir_path = "..\\CV_SHAD_EDGE_BYIM\\GRY\\40x40\\Fold_"
elif train_data_in == "CSHD":
    dir_path = "..\\CV_SHAD_CSHD_BYIM\\GRY\\40x40\\Fold_"
else:
    print("!!!ERROR!!! Invalid data...")
    exit(0)

print("...gabor_model_train.py...")
print("...")
print("...inputs...")
print("...train file      : " + dir_path)
print("...holdback fold   : " + str(test_fold))
print("...l2 penalty      : " + str(l2))

print("...save as         : " + save_dir)
print("...")

x_all = np.zeros((0, 1600))
y_all = np.zeros((0, 2))

for i in range(num_folds):  # num folds
    fold_path = dir_path + str(i + 1) + "\\"
    ld = LoadDataset(fold_path)
    (x, y) = ld.load_dataset()

    x_all = np.append(x_all, x, axis=0)
    y_all = np.append(y_all, y, axis=0)

kfold = KFoldDataset((x_all, y_all), num_folds=5)
(x_train, y_train), (x_test, y_test) =  kfold.make_folds()
(x_train_fold, y_train_fold), (x_test_fold, y_test_fold) = PrepDataset(x_train, y_train, x_test, y_test, test_fold).prep_epoch_data()

y_train_fold = y_train_fold[:,1]
y_test_fold  = y_test_fold[:,1]

model = GaborClass(l2=l2, pool_size= pool_size,strides=strides)
print("...model initialized...")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
print("...model compiled...")
input_shape = x_train_fold.shape
model.build(input_shape)
model.summary()

history = model.fit(x_train_fold,y_train_fold, batch_size =batch_size, epochs = num_epochs, verbose = 1)

print("...EVALUATING MODEL ON TEST BATCH...")
test_scores = model.evaluate(x_test_fold, y_test_fold, verbose=1)
print("Test loss     : ", test_scores[0])
print("Test accuracy : ", test_scores[1])

# save the model
model.save(save_dir)
print("...model saved...")

