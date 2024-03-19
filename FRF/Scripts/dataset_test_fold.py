"""
File: dataset_test_fold.py
Auth: CD @ FGCU
"""

import numpy as np
import scipy.io
from sklearn.model_selection import KFold

class TestFoldDataset:

    # intitializer(s)
    #------------------------------------------------------
    def __init__(self, file_path, test_file, num_img=1000, x_shape=1600, y_shape=2, num_folds=10):
        """
        :param file_path:   root file path
        :param test_file:   test file
        :param num_img:     number of images per file
        :param x_shape:     the shape of the image height * image width
        :param y_shape:     the number of classifications possible
        :param num_folds:   the number of folds to perform on the database
        """
        self.num_img = num_img
        self.test_file = test_file
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.file_path = file_path
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

        return x, y


    # private methods
    def __load_data(self):
        """
        loads the datafiles and create a 2D-tensor

        :returns: A 2D tensor [batch_size, img_height * img_width]
        """

        dataset_all_x = np.ndarray(shape=(0, self.x_shape), dtype=float)
        dataset_all_y = np.ndarray(shape=(0, self.y_shape), dtype=float)

        datafile_x = np.float32(scipy.io.loadmat(self.file_path + str(self.test_file) + ".mat")["image_patches"])
        datafile_y = np.float32(scipy.io.loadmat(self.file_path + str(self.test_file) + ".mat")["category_labels"])

        dataset_all_x = np.append(datafile_x, dataset_all_x, axis=0)
        dataset_all_y = np.append(datafile_y, dataset_all_y, axis=0)

        return dataset_all_x, dataset_all_y
