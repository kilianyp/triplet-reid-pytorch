"""A simple data logger that allows global logging of numpy arrays to various formats.
for machine learning experiments.
"""
import h5py
import os
import sys
import json
from shutil import copyfile
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


#import glob
import re

def get_last_file_number(base_name, path):
    """Gets the number of the last file
    Returns -1 if no file matching the base_name was found in path.
    """

    reg_exp = re.sub("{.*}", "([0-9]+)", base_name)
    # TODO glob?
    files = os.listdir(path)
    reg_exp = re.compile(reg_exp)
    max_value = -1
    for file in files:
        match = re.match(reg_exp, file)
        if match is None:
            continue
        value = int(match.group(1))
        if value > max_value:
            max_value = value

    return max_value

def get_new_numeric_name(base_name, path):
    """Finds all files with a given base name.
    
    Returns a filename counting upwards.

    base_name: contains curly brackets only once {}, place holder for a numeric value
    path: Directory where files are checked.
    """
    max_value = get_last_file_number(base_name, path)
    return base_name.format(max_value + 1)

class Logger(object):
    ARGS_FILE = "args.json"

    @property
    def args_file(self):
        return os.path.join(self.log_dir, self.ARGS_FILE)

    def __init__(self, experiment, output_path, level, origin="pytorch"):
        self.level = level
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        log_dir = os.path.join(output_path, experiment)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        print("Logging to %s" % log_dir)
        
        self.origin = origin
        self.log_dir = log_dir

    def write(self, name, data, dtype):
        raise NotImplemented("The write function for this backend has not been implemented")

    def close(self):
        pass

    def save_args(self, args):
        if self.level == 0:
            print_warning("Warning: Not saving arguments because of logging level 0")
        import subprocess
        label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        args.commit = label
        if os.path.isfile(self.args_file):
            print("Warning!! Args.json will be overwritten!")

        with open(self.args_file, 'w') as file:
            json.dump(vars(args), file, ensure_ascii=False,
                        indent=2, sort_keys=True)

    def restore_args(self):
        with open(self.args_file, 'r') as file:
            return json.load(file)

    def save_description(self, model):
        save_pytorch_description(model)
    
    def _make_model_name(self, iteration):
        return "model_{}".format(iteration)

    def get_model_path(self, iteration):
        path = os.path.join(self.log_dir, self._make_model_name(iteration))
        if not os.path.isfile(path):
            print("Checkpoint not found! {}".format(path))
            return None

        return path

    def save_model_state(self, model, iteration):
        path = os.path.join(self.log_dir, self._make_model_name(iteration)) 
        if self.origin == "pytorch":
            _save_pytorch_model(path, model)
        else:
            raise NotImplementedError("No saving function has been implemented for {}!".format(self.origin))

        print("Saved model to {}".format(path))

    def save_model_file(self, model_name, model_path="models"):
        """Saves the model file to the directory."""
        src = os.path.join(model_path, "{}.py".format(model_name))
        dst = os.path.join(self.log_dir, "{}.py".format(model_name))
        copyfile(src, dst)


class H5Logger(Logger):
    DEFAULT_SIZE = 100
    LOG_FILE = "log_{}.h5"
    def __init__(self, *args):
        super().__init__(*args)
        log_file_name = get_new_numeric_name(self.LOG_FILE, self.log_dir)
        if self.level > 0:
            self.handle = h5py.File(os.path.join(self.log_dir, log_file_name), 'w')
        self.columns = {}

    def write(self, name, data, dtype=None):
        if self.level == 0:
            return

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
        if self.level > 0:
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

def _save_pytorch_model(path, model):
    import torch
    torch.save(model.state_dict(), path)

def save_pytorch_description(model):
    path = os.path.join(LOGGER.log_dir, "model.txt")
    with open(path, 'w') as f:
        print(model, file=f)
