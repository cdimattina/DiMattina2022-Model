"""
File:   load_dataset.py
Author: Josiah Burnham
Date  : 03/21/22
Org:    FGCU Computational Preception Lab
Desc:   Load a dataset stored in .mat files
"""

import numpy as np
import scipy.io
import os


class LoadDataset:
    """
    load a dataset stored in .mat files
    """
    def __init__(self, dir_path, x_shape=1600, y_shape=2):
        """Load a dataset from .mat files and return them as a numpy array
           in the form: (features, labels)

        Args:
            dir_path (string): directory path for train data
            x_shape (int, optional): the shape of the features in the data. Defaults to 1600.
            y_shape (int, optional): the shape of the labels in the data. Defaults to 2.
        """

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dir_path = dir_path


    def load_dataset(self):
        """load the dataset from .mat files

        Returns:
            (numpy array, numpy array): the x and y components to the dataset
        """

        x, y = self.__load_data()
        return (x,y)

    def __load_data(self):
        """loads the datafiles and create a 2D numpy array

        Returns:
            2D numpy array: numpy array with shape: 
                            [number of images, image height * image width]
        """

        dataset_all_x = np.ndarray(shape=(0, self.x_shape), dtype=float)
        dataset_all_y = np.ndarray(shape=(0, self.y_shape), dtype=float)

        files_in_train_dir = [f for f in os.listdir(self.dir_path)
                                if os.path.isfile(os.path.join(self.dir_path, f))]


        for i in files_in_train_dir:
            datafile_x = np.float32(
                scipy.io.loadmat(self.dir_path + i)["image_patches"])

            datafile_y = np.float32(
                                scipy.io.loadmat(self.dir_path + i)["category_labels"])

            dataset_all_x = np.append(datafile_x, dataset_all_x, axis=0)
            dataset_all_y = np.append(datafile_y, dataset_all_y, axis=0)

        return dataset_all_x, dataset_all_y
