"""
File:   filterbank_maker.py
Author: Josiah Burnham (10/2021)
Org:    FGCU
Desc:   Take .mat file dataset batches, and return data, one large tensor
"""

import numpy as np
import scipy.io


class LoadData:
    def __init__(self, file_path, num_files, start_file = 0,  num_img=1000, x_shape=1600, y_shape=2):
        self.file_path = file_path
        self.num_files = num_files
        self.num_img = num_img
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.start_file = start_file

    def load_file_data(self):
        data_set_x = np.ndarray(shape=(0, self.x_shape), dtype=np.float32)
        data_set_y = np.ndarray(shape=(0, self.y_shape), dtype=np.float32)

        for i in range(self.num_files):
            datafile_x = np.float32(scipy.io.loadmat(self.file_path + str(self.start_file) + ".mat")["image_patches"])
            datafile_y = np.float32(scipy.io.loadmat(self.file_path + str(self.start_file) + ".mat")["category_labels"])

            data_set_x = np.append(datafile_x, data_set_x, axis=0)
            data_set_y = np.append(datafile_y, data_set_y, axis=0)

            self.start_file += 1

        return data_set_x, data_set_y
