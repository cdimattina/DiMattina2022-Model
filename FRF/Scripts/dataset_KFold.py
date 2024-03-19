"""
File:   filterbank_maker.py
Author: Josiah Burnham (10/2021)
Org:    FGCU
Desc:   Take .mat file dataset batches, and return data, in two large tensors (train, test) specifying the hold back set
"""

import numpy as np
import scipy.io
from sklearn.model_selection import KFold


class KFoldDataset:
    """
    preform K-Fold crossvalidation on your data
    """

    # intitializer(s)
    #------------------------------------------------------
    def __init__(self, file_path, num_files, num_img=1000, x_shape=1600, y_shape=2, num_folds=10):
        """
        :param file_path:   root file path
        :param num_files:   number of data files
        :param num_img:     number of images per file
        :param x_shape:     the shape of the image height * image width
        :param y_shape:     the number of classifications possible
        :param num_folds:   the number of folds to perform on the database
        """
        self.num_img = num_img
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.file_path = file_path
        self.num_files = num_files
        self.num_folds = num_folds


    # member methods
    #------------------------------------------------------

    # public methods
    def make_dataset(self):
        """
        the main handler for the dataset maker class

        :returns:       A 3D tensor [num_folds, num_images, image_height * image_width]
        """
        x, y = self.__load_data()

        (x_train, y_train), (x_test, y_test) = self.__k_fold(x, y)

        return (x_train, y_train), (x_test, y_test)


    # private methods
    def __load_data(self):
        """
        loads the datafiles and create a 2D-tensor

        :returns: A 2D tensor [batch_size, img_height * img_width]
        """

        dataset_all_x = np.ndarray(shape=(0, self.x_shape), dtype=float)
        dataset_all_y = np.ndarray(shape=(0, self.y_shape), dtype=float)

        for i in range(self.num_files):
            datafile_x = np.float32(
                scipy.io.loadmat(self.file_path + str(i + 1) + ".mat")["image_patches"])

            datafile_y = np.float32(
                scipy.io.loadmat(self.file_path + str(i + 1) + ".mat")["category_labels"]
            )

            dataset_all_x = np.append(datafile_x, dataset_all_x, axis=0)
            dataset_all_y = np.append(datafile_y, dataset_all_y, axis=0)

        return dataset_all_x, dataset_all_y


    def __k_fold(self, dataset_x, dataset_y):
        """
        splits the train and test datasets into x and y batches

        :param dataset_x:       The images dataset
        :param dataset_y:       the labels dataset
        :returns: returns two tuples separated into (x_train, y_train) (x_test, y_test)
        """
        x_train_shape = (self.num_img * self.num_files) - ((self.num_img * self.num_files) // self.num_folds)
        x_test_shape = (self.num_img * self.num_files) // self.num_folds

        x_train_indices, x_test_indices = self.__split_dataset(dataset_x, x_train_shape, x_test_shape)
        y_train_indices, y_test_indices = self.__split_dataset(dataset_y, x_train_shape, x_test_shape)

        x_train_batches = dataset_x[x_train_indices]
        x_test_batches = dataset_x[x_test_indices]

        y_train_batches = dataset_y[y_train_indices]
        y_test_indices = dataset_y[y_test_indices]

        return (x_train_batches, y_train_batches), (x_test_batches, y_test_indices)


    def __split_dataset(self, dataset, train_shape, test_shape):
        """
        splits the dataset into test and train sets, and performs k fold cross validation

        :returns:   two 3D tensors [num_folds, number of test or train  images, img_height * img_width]
        """
        k_fold = KFold(n_splits=self.num_folds)

        train = np.ndarray(shape=(0, train_shape), dtype=np.int32)
        test = np.ndarray(shape=(0, test_shape), dtype=np.int32)

        for train_indices, test_indices in k_fold.split(dataset):
            train_indices = np.expand_dims(train_indices, axis=0)
            test_indices = np.expand_dims(test_indices, axis=0)

            test = np.append(test_indices, test, axis=0)
            train = np.append(train_indices, train, axis=0)
        return train, test

