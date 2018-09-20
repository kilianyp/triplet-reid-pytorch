"""
Model initialization file
"""
import glob
from importlib import import_module
# get all py files in model dir
files = glob.glob("models/[!_\.]*.py")

model_choices = []
for model_file in files:
    if "utils.py" in model_file:
        continue
    model_file = model_file[len("models/"):]
    model_choices.append(model_file[:-len(".py")])

modules = {}
model_parameters = {}
for model in model_choices:
    module = import_module("models.{}".format(model))
    try:
        model_params = getattr(module, "model_parameters")
        model_parameters.update(model_params)
    except Exception as e:
        print(e)
        pass
    modules[model] = module

def build_args(args):
    pass

def get_model(args):
    """
    args: A dict that contains all model parameters."""
    if args["model"] not in model_choices:
        raise RuntimeError("Model not found!")
    init_fn = getattr(modules[args["model"]], args["model"])
    model_parameters.update(args)
    print("Got model from models/{}.py.".format(args["model"]))
    return init_fn(**model_parameters)

