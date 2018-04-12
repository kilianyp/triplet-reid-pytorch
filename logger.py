"""A simple data logger that allows global logging of numpy arrays to various formats.
for machine learning experiments.
"""
import h5py
import os
import sys
import numpy as np
LOGGER = None


def print_warning(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_data_shape(data):
    if hasattr(data, 'shape'):
        dshape = data.shape
    elif type(data) == list or type(data) == tuple:
        dshape = (len(data),)
    else:
        dshape = (1,)

    return dshape
        
class Logger(object):
    def __init__(self, path):
        self.path = path
        if os.path.exists(self.path):
            print_warning("File already exists: {}.".format(self.path))

    def write(self, name, data, dtype):
        raise NotImplemented("The write function for this backend has not been implemented")

    def close(self):
        pass

class H5Logger(Logger):
    DEFAULT_SIZE = 100
    def __init__(self, path):
        super().__init__(path)
        self.handle = h5py.File(self.path, 'w')
        self.columns = {}

    def write(self, name, data, dtype=None):
        if name not in self.columns:
            if dtype == None:
                dtype = data.dtype
            dshape = get_data_shape(data)

            maxshape = tuple([None] + list(dshape))
            shape = tuple([self.DEFAULT_SIZE] + list(dshape))
            dataset = self.handle.create_dataset(name, shape=shape, maxshape=maxshape,
                                                 dtype=dtype)
            position = 0
            self.columns[name] = [dataset, position, self.DEFAULT_SIZE] # dataset handle, position, maxsize
        else:
            dataset, position, maxsize = self.columns[name]
            # position starts at 0
            if position + 1 > maxsize:
                # increase dataset by max_size
                maxsize = maxsize + self.DEFAULT_SIZE
                dshape = get_data_shape(data)
                shape = tuple([maxsize] + list(dshape))
                dataset.resize(shape)
                print("Increased datset size to {}".format(maxsize))
                self.columns[name][2] = maxsize
        dataset[position] = data
        self.columns[name][1] += 1
    
    def close(self):
        self.handle.close()

def create_logger(path, type):
    global LOGGER
    if type == "h5":
        logger = H5Logger(path)
    else:
        raise NotImplemented("Type is not implemented: {}".format(type))
    
    LOGGER = logger
    return logger

def write(name, data, dtype=None):
    if LOGGER is None:
        raise RuntimeError("No logger has been created!")
    LOGGER.write(name, data, dtype)

def close():
    LOGGER.close()
