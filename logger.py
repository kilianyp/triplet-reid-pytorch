"""A simple data logger that allows global logging of numpy arrays to various formats.
for machine learning experiments.
"""
import h5py
import os
import sys
import numpy as np
import json
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
    ARGS_FILE = "args.json"
    def __init__(self, experiment, output_path, level):
        self.level = level
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        log_dir = os.path.join(output_path, experiment)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        print("Logging to %s" % log_dir)
        
        self.log_dir = log_dir
        self.args_file = os.path.join(self.log_dir, self.ARGS_FILE)

    def write(self, name, data, dtype):
        raise NotImplemented("The write function for this backend has not been implemented")

    def close(self):
        pass

    def save_args(self, args):
        if self.level == 0:
            print_warning("Warning: Not saving arguments because of logging level 0")

        with open(self.args_file, 'w') as file:
            json.dump(vars(args), file, ensure_ascii=False,
                        indent=2, sort_keys=True)

    def restore_args(self):
        with open(self.args_file, 'r') as file:
            return json.load(file)


class H5Logger(Logger):
    DEFAULT_SIZE = 100
    LOG_FILE = "log.h5"
    def __init__(self, *args):
        super().__init__(*args)
        self.handle = h5py.File(os.path.join(self.log_dir, self.LOG_FILE), 'w')
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
                print("Increased datset {} size to {}".format(name, maxsize))
                self.columns[name][2] = maxsize
        dataset[position] = data
        self.columns[name][1] += 1
    
    def close(self):
        self.handle.close()

def create_logger(type, *args, **kwargs):
    global LOGGER
    if type == "h5":
        logger = H5Logger(*args, **kwargs)
    else:
        raise NotImplemented("Type is not implemented: {}".format(type))
    
    LOGGER = logger
    return logger

def write(name, data, dtype=None):
    if LOGGER is None:
        raise RuntimeError("No logger has been created!")
    LOGGER.write(name, data, dtype)

def save_pytorch_model(model, iteration):
    import torch
    torch.save(model.state_dict(), os.path.join(LOGGER.log_dir, "model_{}".format(iteration)))
    print("Saved model to {}".format(LOGGER.log_dir))
